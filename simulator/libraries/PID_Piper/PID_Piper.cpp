#include "PID_Piper.h"
#include "../../ArduCopter/O_PID_Piper.h"
#include <AP_HAL/AP_HAL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <fstream>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <ctime>
#include <vector>

// ML input logging (记录喂给 ML 的输入，供人工离线判断是否存在攻击)
static std::ofstream ml_input_ofs;
static std::once_flag ml_input_init_flag;
static std::mutex ml_input_mutex;
static std::string ml_input_filename = "/pid-piper/simulator/Data_Piper_ML_Inputs.csv";

// Attack wide-table logging (mode1 仿真每次产出一个 attack+时间 宽表)
static std::ofstream training_wide_ofs;
static std::once_flag training_wide_init_flag;
static std::mutex training_wide_mutex;
static std::string training_wide_filename = "/pid-piper/simulator/Data_Piper_Training_Wide.csv";

namespace {
static constexpr size_t kGateFeatureCount = 22;
static constexpr float kGateZClip = 8.0f;
static constexpr size_t kMode4GateSequenceLength = 100;
static constexpr size_t kUserModelTimeSteps = 1;
static constexpr size_t kUserModelFeatureCount = 18;
static constexpr float kUserModelOutputScale = 1.0e-3f;
static constexpr float kUserRollPitchAbsMaxRad = 2.0f;
static constexpr float kUserYawAbsMaxRad = 3.5f;
// Strategy mode quick override for easier local testing/editing.
// -1: use runtime parameter value (PIPER_MODE)
//  0: force pure PID
//  1: force original ML only
//  2: force hard switch (PID until recovery, then original ML)
//  3: force legacy gate-NN + original ML fusion
//  4: force LSTM gate-NN + original ML fusion
static constexpr int8_t kStrategyModeOverride = -1;

static void append_unique(std::vector<std::string> &out, const std::string &value)
{
	if (value.empty()) {
		return;
	}
	if (std::find(out.begin(), out.end(), value) == out.end()) {
		out.push_back(value);
	}
}

static bool file_exists(const std::string &path)
{
	std::ifstream ifs(path.c_str());
	return ifs.good();
}

static std::string join_path(const std::string &base, const std::string &leaf)
{
	if (base.empty()) {
		return leaf;
	}
	if (base.back() == '/' || base.back() == '\\') {
		return base + leaf;
	}
	return base + "/" + leaf;
}

static std::vector<std::string> output_file_candidates(const char *env_file_name, const char *leaf_name)
{
	std::vector<std::string> candidates;

	const char *env_file = std::getenv(env_file_name);
	if (env_file != nullptr) {
		append_unique(candidates, std::string(env_file));
	}

	const char *env_output_dir = std::getenv("PID_PIPER_OUTPUT_DIR");
	if (env_output_dir != nullptr) {
		append_unique(candidates, join_path(std::string(env_output_dir), std::string(leaf_name)));
	}

	const char *env_root = std::getenv("PID_PIPER_ROOT");
	if (env_root != nullptr) {
		append_unique(candidates, join_path(join_path(std::string(env_root), "simulator"), std::string(leaf_name)));
	}

	append_unique(candidates, join_path("/pid-piper/simulator", std::string(leaf_name)));
	append_unique(candidates, join_path("../", std::string(leaf_name)));
	append_unique(candidates, std::string(leaf_name));

	return candidates;
}

static std::string resolve_model_path(const char *model_leaf_name)
{
	std::vector<std::string> candidates;

	const char *env_model_dir = std::getenv("PID_PIPER_MODEL_DIR");
	if (env_model_dir != nullptr) {
		append_unique(candidates, join_path(std::string(env_model_dir), std::string(model_leaf_name)));
	}

	const char *env_root = std::getenv("PID_PIPER_ROOT");
	if (env_root != nullptr) {
		append_unique(candidates, join_path(join_path(std::string(env_root), "simulator/libraries/PID_Piper/models"), std::string(model_leaf_name)));
	}

	append_unique(candidates, join_path("/pid-piper/simulator/libraries/PID_Piper/models", std::string(model_leaf_name)));
	append_unique(candidates, join_path("../libraries/PID_Piper/models", std::string(model_leaf_name)));
	append_unique(candidates, join_path("libraries/PID_Piper/models", std::string(model_leaf_name)));
	append_unique(candidates, join_path("simulator/libraries/PID_Piper/models", std::string(model_leaf_name)));

	for (const std::string &path : candidates) {
		if (file_exists(path)) {
			return path;
		}
	}

	return join_path("/pid-piper/simulator/libraries/PID_Piper/models", std::string(model_leaf_name));
}

static bool open_append_with_fallback(std::ofstream &ofs, std::string &selected_path,
		const std::vector<std::string> &candidates)
{
	for (const std::string &path : candidates) {
		ofs.open(path.c_str(), std::ios::out | std::ios::app);
		if (ofs.is_open()) {
			selected_path = path;
			return true;
		}
		ofs.clear();
	}
	return false;
}

static const std::array<float, kGateFeatureCount> kRollGateMean = {
	153.5175476f, 14.37909794f, -1.401101589f, 0.001089086174f, -0.001137534855f, 0.000816375541f,
	23072.91992f, -1814.68689f, 5002.120117f, 377.6418762f, 0.002940058708f, 0.004001242109f,
	906.2440796f, -37.04817581f, -5.92607832f, 2.31624341f, 20.28152275f, -1440.270752f,
	0.01294364035f, 0.01495643519f, 0.022956267f, 0.022956267f
};

static const std::array<float, kGateFeatureCount> kRollGateStd = {
	356.2026062f, 104.2925568f, 15.44612026f, 0.1417743266f, 0.104490906f, 0.01488449331f,
	17723.20508f, 1221.211182f, 12.1380167f, 189.6291046f, 0.001519187004f, 0.01327415369f,
	534.9769897f, 145.924408f, 62.42513657f, 28.38473892f, 257.8906555f, 688.9980469f,
	0.04377181828f, 0.05011707544f, 0.02888619341f, 0.02888618968f
};

static const std::array<float, kGateFeatureCount> kPitchGateMean = {
	153.5175476f, 14.37909794f, -1.401101589f, 0.001089086174f, -0.001137534855f, 0.000816375541f,
	23072.91992f, -1814.68689f, 5002.120117f, 377.6418762f, 0.002940058708f, 0.004001242109f,
	906.2440796f, -37.04817581f, -5.92607832f, 2.31624341f, 20.28152275f, -1440.270752f,
	-0.1646319479f, -0.2747250199f, 0.1101742759f, 0.110174261f
};

static const std::array<float, kGateFeatureCount> kPitchGateStd = {
	356.2026062f, 104.2925568f, 15.44612026f, 0.1417743266f, 0.104490906f, 0.01488449331f,
	17723.20508f, 1221.211182f, 12.1380167f, 189.6291046f, 0.001519187004f, 0.01327415369f,
	534.9769897f, 145.924408f, 62.42513657f, 28.38473892f, 257.8906555f, 688.9980469f,
	0.3108014464f, 0.1297312379f, 0.1849065125f, 0.1849065125f
};

static const std::array<float, kGateFeatureCount> kYawGateMean = {
	153.5145874f, 14.37962151f, -1.401137114f, 0.001089095487f, -0.001137556625f, 0.0008163680905f,
	23072.34766f, -1814.647583f, 5002.120605f, 377.6398315f, 0.002940068953f, 0.004001267254f,
	906.2377319f, -37.04740906f, -5.92618084f, 2.316282034f, 20.28137207f, -1440.264038f,
	0.08757957816f, -0.1444928944f, 0.2324033827f, 0.2324033976f
};

static const std::array<float, kGateFeatureCount> kYawGateStd = {
	356.2049866f, 104.2934036f, 15.44625282f, 0.1417755634f, 0.104491815f, 0.01488462277f,
	17722.83398f, 1221.185547f, 12.13812065f, 189.6301575f, 0.00151919818f, 0.01327426825f,
	534.9795532f, 145.9255829f, 62.42568207f, 28.38498497f, 257.8929138f, 689.0021973f,
	0.6280271411f, 0.01250074431f, 0.6209352612f, 0.6209352016f
};

static const std::array<float, kGateFeatureCount> kRollMode4GateMean = {
	4.77291679f, 5.97398186f, -0.00255395798f, 0.00199813559f, 0.000485316588f, 0.000533120416f,
	16660.1621f, -533.55957f, 5004.89209f, 376.308899f, 0.0200884249f, 0.0328402705f,
	1009.15845f, 134.270554f, 39.6361771f, 44.2064476f, 271.669159f, -1398.32422f,
	0.496875972f, 0.0477620512f, 0.451010942f, 0.451010942f
};

static const std::array<float, kGateFeatureCount> kRollMode4GateStd = {
	6.14559937f, 2.13424301f, 0.0756343231f, 0.0948762968f, 0.0790506527f, 0.0808886141f,
	7909.93066f, 1220.08191f, 2.52548099f, 53.4715042f, 0.0222751796f, 0.042932421f,
	435.342743f, 133.213669f, 123.87265f, 31.6160183f, 151.904678f, 186.024445f,
	0.176557839f, 0.0269969702f, 0.167882144f, 0.167882144f
};

static const std::array<float, kGateFeatureCount> kPitchMode4GateMean = {
	2.33123803f, 3.95026016f, -0.033393085f, 0.00225008815f, -0.000182807911f, 0.000787730154f,
	14219.3809f, -600.08197f, 5000.104f, 313.0513f, 0.0144155128f, 0.023472622f,
	715.186584f, 111.764359f, -24.7538204f, 38.5849648f, 169.424133f, -1271.35559f,
	-0.207904354f, -0.221688509f, 0.533418834f, 0.533418834f
};

static const std::array<float, kGateFeatureCount> kPitchMode4GateStd = {
	9.50864506f, 2.59355211f, 0.210172445f, 0.0989475325f, 0.0751343444f, 0.0692561418f,
	9967.84668f, 1197.41687f, 16.5537033f, 127.939522f, 0.0191432796f, 0.035804268f,
	738.033691f, 156.35524f, 195.159424f, 46.3132362f, 249.007492f, 318.834229f,
	0.630553484f, 0.0560688563f, 0.230906785f, 0.230906785f
};

static const std::array<float, kGateFeatureCount> kYawMode4GateMean = {
	2.52813196f, 5.16095686f, -0.0236582123f, 0.00237266393f, -0.00283661694f, -3.31263145e-05f,
	13856.7744f, -593.813843f, 5002.26172f, 336.873291f, 0.0156842358f, 0.025167983f,
	830.082581f, 135.697327f, 3.96738529f, 42.6263504f, 225.873627f, -1318.7113f,
	0.235223368f, -0.137885422f, 0.373302907f, 0.373302907f
};

static const std::array<float, kGateFeatureCount> kYawMode4GateStd = {
	7.62172461f, 2.85497761f, 0.19142881f, 0.0858891383f, 0.0774882212f, 0.0712830946f,
	8806.47168f, 1097.28857f, 15.9331236f, 109.102989f, 0.0194684919f, 0.0382159501f,
	627.74469f, 132.799377f, 164.463913f, 33.7246704f, 193.052628f, 276.735596f,
	0.610649168f, 0.0080497358f, 0.60460937f, 0.60460937f
};

static std::vector<float> build_gate_input(const PID_Piper &state, float y_pid, float y_ml,
		float residual_value, const std::array<float, kGateFeatureCount> &mean,
		const std::array<float, kGateFeatureCount> &std)
{
	const std::array<float, kGateFeatureCount> raw = {
		state.acc.x, state.acc.y, state.acc.z,
		state.gyro.x, state.gyro.y, state.gyro.z,
		state.pos.x, state.pos.y, state.pos.z,
		state.gpsVel,
		state.ahrsRP, state.ahrsYaw,
		state.posVarH, state.posVarV,
		state.velVarX, state.velVarY,
		static_cast<float>(state.navRoll), static_cast<float>(state.navPitch),
		y_pid, y_ml, residual_value, fabsf(y_ml - y_pid)
	};

	std::vector<float> normalized;
	normalized.reserve(kGateFeatureCount);
	for (size_t i = 0; i < kGateFeatureCount; ++i) {
		const float std_i = (fabsf(std[i]) < 1.0e-6f) ? 1.0f : std[i];
		const float z = (raw[i] - mean[i]) / std_i;
		normalized.push_back(constrain_float(z, -kGateZClip, kGateZClip));
	}
	return normalized;
}

struct GateSequenceBuffer {
	std::array<std::array<float, kGateFeatureCount>, kMode4GateSequenceLength> rows{};
	size_t count = 0;
	size_t next = 0;

	void clear()
	{
		count = 0;
		next = 0;
	}

	void push(const std::vector<float> &row)
	{
		for (size_t i = 0; i < kGateFeatureCount; ++i) {
			rows[next][i] = row[i];
		}
		next = (next + 1) % kMode4GateSequenceLength;
		if (count < kMode4GateSequenceLength) {
			++count;
		}
	}

	bool ready() const
	{
		return count >= kMode4GateSequenceLength;
	}

	std::vector<float> to_tensor_values() const
	{
		std::vector<float> values;
		values.reserve(kMode4GateSequenceLength * kGateFeatureCount);
		const size_t start = (count < kMode4GateSequenceLength) ? 0 : next;
		for (size_t step = 0; step < kMode4GateSequenceLength; ++step) {
			const size_t row_index = (start + step) % kMode4GateSequenceLength;
			values.insert(values.end(), rows[row_index].begin(), rows[row_index].end());
		}
		return values;
	}
};

static GateSequenceBuffer &roll_mode4_gate_buffer()
{
	static GateSequenceBuffer buffer;
	return buffer;
}

static GateSequenceBuffer &pitch_mode4_gate_buffer()
{
	static GateSequenceBuffer buffer;
	return buffer;
}

static GateSequenceBuffer &yaw_mode4_gate_buffer()
{
	static GateSequenceBuffer buffer;
	return buffer;
}

static void reset_mode4_gate_buffers()
{
	roll_mode4_gate_buffer().clear();
	pitch_mode4_gate_buffer().clear();
	yaw_mode4_gate_buffer().clear();
}

static const fdeep::model &roll_gate_model()
{
	static const fdeep::model model = fdeep::load_model(resolve_model_path("roll-gate-nn.json"));
	return model;
}

static const fdeep::model &pitch_gate_model()
{
	static const fdeep::model model = fdeep::load_model(resolve_model_path("pitch-gate-nn.json"));
	return model;
}

static const fdeep::model &yaw_gate_model()
{
	static const fdeep::model model = fdeep::load_model(resolve_model_path("yaw-gate-nn.json"));
	return model;
}

static const fdeep::model &roll_mode4_gate_model()
{
	static const fdeep::model model = fdeep::load_model(resolve_model_path("roll-gate-lstm-mode4.json"));
	return model;
}

static const fdeep::model &pitch_mode4_gate_model()
{
	static const fdeep::model model = fdeep::load_model(resolve_model_path("pitch-gate-lstm-mode4.json"));
	return model;
}

static const fdeep::model &yaw_mode4_gate_model()
{
	static const fdeep::model model = fdeep::load_model(resolve_model_path("yaw-gate-lstm-mode4.json"));
	return model;
}

static const fdeep::model &roll_user_ml_model()
{
	static const fdeep::model model = []() {
		const std::string path = resolve_model_path("roll_user_fdeep.json");
		try {
			fdeep::model m = fdeep::load_model(path);
			::fprintf(stderr, "[PID_PIPER] loaded user model: %s\n", path.c_str());
			return m;
		} catch (const std::exception& ex) {
			throw std::runtime_error(std::string("load failed for ") + path + ": " + ex.what());
		}
	}();
	return model;
}

static const fdeep::model &pitch_user_ml_model()
{
	static const fdeep::model model = []() {
		const std::string path = resolve_model_path("pitch_user_fdeep.json");
		try {
			fdeep::model m = fdeep::load_model(path);
			::fprintf(stderr, "[PID_PIPER] loaded user model: %s\n", path.c_str());
			return m;
		} catch (const std::exception& ex) {
			throw std::runtime_error(std::string("load failed for ") + path + ": " + ex.what());
		}
	}();
	return model;
}

static const fdeep::model &yaw_user_ml_model()
{
	static const fdeep::model model = []() {
		const std::string path = resolve_model_path("yaw_user_fdeep.json");
		try {
			fdeep::model m = fdeep::load_model(path);
			::fprintf(stderr, "[PID_PIPER] loaded user model: %s\n", path.c_str());
			return m;
		} catch (const std::exception& ex) {
			throw std::runtime_error(std::string("load failed for ") + path + ": " + ex.what());
		}
	}();
	return model;
}

static float predict_gate_alpha(const fdeep::model &model, const std::vector<float> &normalized_input)
{
	const float alpha = model.predict_single_output({fdeep::tensor(fdeep::tensor_shape(kGateFeatureCount), normalized_input)});
	if (!std::isfinite(alpha)) {
		throw std::runtime_error("non-finite gate alpha");
	}
	return constrain_float(alpha, 0.0f, 1.0f);
}

static float predict_lstm_gate_alpha(const fdeep::model &model, GateSequenceBuffer &buffer,
		const std::vector<float> &normalized_input)
{
	buffer.push(normalized_input);
	if (!buffer.ready()) {
		return 0.0f;
	}

	std::vector<float> sequence_values = buffer.to_tensor_values();
	const float alpha = model.predict_single_output({
		fdeep::tensor(fdeep::tensor_shape(kMode4GateSequenceLength, kGateFeatureCount), sequence_values)
	});
	if (!std::isfinite(alpha)) {
		throw std::runtime_error("non-finite LSTM gate alpha");
	}
	return constrain_float(alpha, 0.0f, 1.0f);
}

static void log_gate_fallback_warning(const char* reason)
{
	static uint64_t last_warn_us = 0;
	const uint64_t now_us = AP_HAL::micros64();
	if (last_warn_us != 0 && (now_us - last_warn_us) < 2000000ULL) {
		return;
	}
	last_warn_us = now_us;
	::fprintf(stderr, "[PID_PIPER] gate inference failed, fallback to ML: %s\n", reason);
}

static void log_user_ml_warning(const char *axis_name, const char *reason)
{
	static uint64_t last_warn_us = 0;
	const uint64_t now_us = AP_HAL::micros64();
	if (last_warn_us != 0 && (now_us - last_warn_us) < 2000000ULL) {
		return;
	}
	last_warn_us = now_us;
	::fprintf(stderr, "[PID_PIPER] user ML %s issue: %s\n", axis_name, reason);
}

static float sanitize_user_ml_output(const char *axis_name, float value, float clamp_abs)
{
	if (!std::isfinite(value)) {
		log_user_ml_warning(axis_name, "non-finite output, substituting 0");
		return 0.0f;
	}

	const float clamped = constrain_float(value, -clamp_abs, clamp_abs);
	if (fabsf(clamped - value) > 1.0e-6f) {
		log_user_ml_warning(axis_name, "output clamped to safe range");
	}
	return clamped;
}
}

static double pidpiper_now_seconds()
{
	return static_cast<double>(AP_HAL::micros64()) * 1.0e-6;
}

static std::string make_attack_wide_filename()
{
	const auto now = std::chrono::system_clock::now();
	const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
	std::tm tm_now{};
#if defined(_WIN32) || defined(_WIN64)
	localtime_s(&tm_now, &now_time);
#else
	localtime_r(&now_time, &tm_now);
#endif
	char time_buffer[32];
	std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", &tm_now);
	const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
	char filename[64];
	std::snprintf(filename, sizeof(filename), "attack_%s_%03lld.csv", time_buffer, static_cast<long long>(millis));
	return std::string(filename);
}

static void init_ml_input_file_pidpiper()
{
	std::lock_guard<std::mutex> guard(ml_input_mutex);
	if (ml_input_ofs.is_open()) return;
	const std::vector<std::string> candidates = output_file_candidates("PID_PIPER_ML_INPUT_FILE", "Data_Piper_ML_Inputs.csv");
	open_append_with_fallback(ml_input_ofs, ml_input_filename, candidates);
	if (!ml_input_ofs.is_open()) return;
	bool empty = true;
	{
		std::ifstream ifs(ml_input_filename.c_str());
		if (ifs.good()) {
			ifs.seekg(0, std::ios::end);
			auto pos = ifs.tellg();
			if (pos > 0) empty = false;
		}
	}
	if (empty) {
		ml_input_ofs << "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,pos_x,pos_y,pos_z,gpsVel,ahrsRP,ahrsYaw,posVarH,posVarV,velVarX,velVarY,navRoll,navPitch,navYaw,angle_type\n";
		ml_input_ofs.flush();
	}
}

static void log_ml_input_vec(const std::vector<float> &v, const char *angle_name)
{
	std::call_once(ml_input_init_flag, init_ml_input_file_pidpiper);
	std::lock_guard<std::mutex> guard(ml_input_mutex);
	if (!ml_input_ofs.is_open()) return;
	ml_input_ofs << pidpiper_now_seconds();
	for (auto &val : v) ml_input_ofs << "," << val;
	ml_input_ofs << "," << angle_name << "\n";
	ml_input_ofs.flush();
}

static void init_training_wide_file_pidpiper()
{
	std::lock_guard<std::mutex> guard(training_wide_mutex);
	if (training_wide_ofs.is_open()) return;
	const std::string attack_leaf = make_attack_wide_filename();
	const std::vector<std::string> candidates = output_file_candidates("PID_PIPER_TRAINING_WIDE_FILE", attack_leaf.c_str());
	open_append_with_fallback(training_wide_ofs, training_wide_filename, candidates);
	if (!training_wide_ofs.is_open()) return;
	bool empty = true;
	{
		std::ifstream ifs(training_wide_filename.c_str());
		if (ifs.good()) {
			ifs.seekg(0, std::ios::end);
			auto pos = ifs.tellg();
			if (pos > 0) empty = false;
		}
	}
	if (empty) {
		training_wide_ofs << "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,pos_x,pos_y,pos_z,gpsVel,ahrsRP,ahrsYaw,posVarH,posVarV,velVarX,velVarY,navRoll,navPitch,navYaw,angle_type,y_pid,y_ml,residual,attack_label,recovery_mode,strategy_mode,alpha,y_fused,y_selected\n";
		training_wide_ofs.flush();
	}
}

static void log_training_wide_row_pidpiper(const PID_Piper &state, const char *angle_name,
		float y_pid, float y_ml, float residual_value, float alpha_value,
		float y_fused, float y_selected, bool recovery_mode)
{
	std::call_once(training_wide_init_flag, init_training_wide_file_pidpiper);
	std::lock_guard<std::mutex> guard(training_wide_mutex);
	if (!training_wide_ofs.is_open()) return;
	training_wide_ofs << pidpiper_now_seconds() << ","
		<< state.acc.x << "," << state.acc.y << "," << state.acc.z << ","
		<< state.gyro.x << "," << state.gyro.y << "," << state.gyro.z << ","
		<< state.pos.x << "," << state.pos.y << "," << state.pos.z << ","
		<< state.gpsVel << "," << state.ahrsRP << "," << state.ahrsYaw << ","
		<< state.posVarH << "," << state.posVarV << ","
		<< state.velVarX << "," << state.velVarY << ","
		<< state.navRoll << "," << state.navPitch << "," << state.navYaw << ","
		<< angle_name << ","
		<< y_pid << "," << y_ml << "," << residual_value << ","
		<< state.attackLabel << "," << (recovery_mode ? 1 : 0) << "," << static_cast<int>(state.strategyMode) << ","
		<< alpha_value << "," << y_fused << "," << y_selected << "\n";
	training_wide_ofs.flush();
}

static void log_training_wide_rows_pidpiper(const PID_Piper &state, const Vector3f &selected, bool recovery_mode)
{
	log_training_wide_row_pidpiper(state, "roll", state.y_PID.x, state.y_ML.x, state.residual[0], state.alpha_gate.x, state.y_fused.x, selected.x, recovery_mode);
	log_training_wide_row_pidpiper(state, "pitch", state.y_PID.y, state.y_ML.y, state.residual[1], state.alpha_gate.y, state.y_fused.y, selected.y, recovery_mode);
	log_training_wide_row_pidpiper(state, "yaw", state.y_PID.z, state.y_ML.z, state.residual[2], state.alpha_gate.z, state.y_fused.z, selected.z, recovery_mode);
}

/*
void PID_Piper::PID_Piper(const PID_Piper &obj)
{
		acc = obj.acc;
		gyro = obj.gyro;
		pos = obj.gyro;
		mag = obj.mag;
		gpsVel = obj.gpsVel;
		ahrsRP = obj.ahrsRP;
		ahrsYaw = obj.ahrsYaw;
		posVarH = obj.posVarH;
		posVarV = obj.posVarV;
		velVarX = obj.velVarX;
		velVarY = obj.velVarY;
		navRoll = obj.navRoll;
		navPitch = obj.navPitch;
}
*/
void PID_Piper::getAirSpeed(float _airspeed)
{
		airspeed = _airspeed;
}

void PID_Piper::getPosControlXY(float _accX, float _accY, float _accZ,
		float _gyroX, float _gyroY, float _gyroZ,
		float _posX, float _posY, float _posZ,
		float _velX, float _ahrsRP, float _ahrsYaw,
		float _errorX, float _errorY, float _velErrorX, float _velErrorY,
		int _navRoll, int _navPitch, int _navYaw, int _attackLabel)
{
		// Keep acceleration in m/s^2 so attack logs match the clean wide-table output.
		acc.x = _accX * 0.01f;
		acc.y = _accY * 0.01f;
		acc.z = _accZ * 0.01f;
		gyro.x = _gyroX;
		gyro.y = _gyroY;
		gyro.z = _gyroZ;
		pos.x = _posX;
		pos.y = _posY;
		pos.z = _posZ;
		gpsVel = _velX;
		ahrsRP = _ahrsRP;
		ahrsYaw = _ahrsYaw;
		posVarH = _errorX;
		posVarV = _errorY;
		velVarX = _velErrorX;
		velVarY = _velErrorY;
		navRoll = _navRoll;
		navPitch = _navPitch;
		navYaw = _navYaw;
		attackLabel = (_attackLabel != 0) ? 1 : 0;
}

void PID_Piper::getPosControlZ(float _accZ)
{
	acc.z = _accZ * 0.01f;
}

void PID_Piper::getMagnetometerData(float _x, float _y, float _z)
{
	mag.x = _x;
	mag.y = _y;
	mag.z = _z;
}

Vector3f PID_Piper::getEulerAngle()
{
	std::vector<float> ml_input = {acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z, pos.x, pos.y, pos.z, gpsVel, ahrsRP, ahrsYaw, posVarH, posVarV, velVarX, velVarY, (float)navRoll, (float)navPitch, (float)navYaw};
	if (strategyMode != 1) {
		log_ml_input_vec(ml_input, "roll");
	}
	y_ML.x = _lstm.getRollAngle(acc, gyro, pos, mag, gpsVel, ahrsRP, ahrsYaw, posVarH, posVarV, velVarX, velVarY, navRoll, navPitch, airspeed);
	if (strategyMode != 1) {
		log_ml_input_vec(ml_input, "pitch");
	}
	y_ML.y = _lstm.getPitchAngle(acc, gyro, pos, mag, gpsVel, ahrsRP, ahrsYaw, posVarH, posVarV, velVarX, velVarY, navRoll, navPitch, airspeed);
	if (strategyMode != 1) {
		log_ml_input_vec(ml_input, "yaw");
	}
	y_ML.z = _lstm.getYawAngle(acc, gyro, pos, mag, gpsVel, ahrsRP, ahrsYaw, posVarH, posVarV, velVarX, velVarY, navRoll, navPitch, airspeed);

	return y_ML;
}

Vector3f PID_Piper::getEulerAngleUser()
{
	static bool user_ml_infer_ok_logged = false;
	const std::vector<float> ml_input = {
		acc.x, acc.y, acc.z,
		gyro.x, gyro.y, gyro.z,
		pos.x, pos.y, pos.z,
		gpsVel, ahrsRP, ahrsYaw,
		posVarH, posVarV,
		velVarX, velVarY,
		(float)navRoll, (float)navPitch
	};

	log_ml_input_vec(ml_input, "roll_user");
	const fdeep::tensor user_input_tensor(fdeep::tensor_shape(kUserModelFeatureCount), ml_input);
	float roll_user_raw = 0.0f;
	try {
		roll_user_raw = roll_user_ml_model().predict_single_output({user_input_tensor});
	} catch (const std::exception& ex) {
		log_user_ml_warning("roll", ex.what());
	}
	log_ml_input_vec(ml_input, "pitch_user");
	float pitch_user_raw = 0.0f;
	try {
		pitch_user_raw = pitch_user_ml_model().predict_single_output({user_input_tensor});
	} catch (const std::exception& ex) {
		log_user_ml_warning("pitch", ex.what());
	}
	log_ml_input_vec(ml_input, "yaw_user");
	float yaw_user_raw = 0.0f;
	try {
		yaw_user_raw = yaw_user_ml_model().predict_single_output({user_input_tensor});
	} catch (const std::exception& ex) {
		log_user_ml_warning("yaw", ex.what());
	}

	const float roll_user = sanitize_user_ml_output("roll", roll_user_raw * kUserModelOutputScale, kUserRollPitchAbsMaxRad);
	const float pitch_user = sanitize_user_ml_output("pitch", pitch_user_raw * kUserModelOutputScale, kUserRollPitchAbsMaxRad);
	const float yaw_user = sanitize_user_ml_output("yaw", yaw_user_raw * kUserModelOutputScale, kUserYawAbsMaxRad);

	y_ML.x = roll_user;
	y_ML.y = pitch_user;
	y_ML.z = yaw_user;
	if (!user_ml_infer_ok_logged) {
		::fprintf(stderr, "[PID_PIPER] user ML inference active (mode2 path)\n");
		user_ml_infer_ok_logged = true;
	}
	return y_ML;
}

void PID_Piper::cusum(double error[3])
{
	residual[0] = error[0];
	residual[1] = error[1];
	residual[2] = error[2];

	for(int i =0;i<3;i++)
	{
		if(delta[i]+residual[i]-b[i]<0)
			delta[i]=0;
		else
			delta[i]+=(residual[i]-b[i]);
	}

	for(int i=0;i<3;i++)
	{
		if(delta[i]>threshold[i])
			recoveryMode = true;
	}

}

bool PID_Piper::checkSwitchControl()
{
	if(abs(residual[0] - b[0]) < 0.1 && abs(residual[1] - b[1]) < 0.1 && abs(residual[2] - b[2]) < 0.1)
	{
		recoveryMode = false;
	}
	return recoveryMode;
}

void PID_Piper::setStrategyMode(uint8_t _mode)
{
	if (kStrategyModeOverride >= 0 && kStrategyModeOverride <= 4) {
		strategyMode = static_cast<uint8_t>(kStrategyModeOverride);
		return;
	}
	strategyMode = (_mode <= 4) ? _mode : 0;
}

Vector3f PID_Piper::recoveryMonitor()
{
	alpha_gate = Vector3f(0.0f, 0.0f, 0.0f);
	y_fused = y_PID;
	residual[0] = 0.0;
	residual[1] = 0.0;
	residual[2] = 0.0;
	if (strategyMode != 4) {
		reset_mode4_gate_buffers();
	}

	if (strategyMode == 0) {
		// mode0: pure PID, no ML inference in the control path
		recoveryMode = false;
		y_ML = y_PID;
		log_training_wide_rows_pidpiper(*this, y_PID, recoveryMode);
		return y_PID;
	}

	y_ML = getEulerAngle();

	// difference between original ML and PID
	residual[0] = abs(y_ML.x - y_PID.x);
	residual[1] = abs(y_ML.y - y_PID.y);
	residual[2] = abs(y_ML.z - y_PID.z);

	if (strategyMode == 1) {
		// mode1: always use original ML
		recoveryMode = true;
		y_fused = y_ML;
		log_training_wide_rows_pidpiper(*this, y_ML, recoveryMode);
		return y_ML;
	}

	if (strategyMode == 2) {
		// mode2: hard switch between PID and original ML based on the residual detector
		cusum(residual);

		if (recoveryMode) {
			recoveryMode = checkSwitchControl();
		}

		const Vector3f selected = recoveryMode ? y_ML : y_PID;
		y_fused = selected;
		log_training_wide_rows_pidpiper(*this, selected, recoveryMode);
		return selected;
	}

	if (strategyMode == 3) {
		// mode3: legacy one-frame gate-NN + original ML fusion
		recoveryMode = true;
		bool fusion_ready = false;

		try {
			const std::vector<float> roll_gate_input = build_gate_input(*this, y_PID.x, y_ML.x, residual[0], kRollGateMean, kRollGateStd);
			const std::vector<float> pitch_gate_input = build_gate_input(*this, y_PID.y, y_ML.y, residual[1], kPitchGateMean, kPitchGateStd);
			const std::vector<float> yaw_gate_input = build_gate_input(*this, y_PID.z, y_ML.z, residual[2], kYawGateMean, kYawGateStd);

			alpha_gate.x = predict_gate_alpha(roll_gate_model(), roll_gate_input);
			alpha_gate.y = predict_gate_alpha(pitch_gate_model(), pitch_gate_input);
			alpha_gate.z = predict_gate_alpha(yaw_gate_model(), yaw_gate_input);

			y_fused.x = y_PID.x + alpha_gate.x * (y_ML.x - y_PID.x);
			y_fused.y = y_PID.y + alpha_gate.y * (y_ML.y - y_PID.y);
			y_fused.z = y_PID.z + alpha_gate.z * (y_ML.z - y_PID.z);

			if (!std::isfinite(y_fused.x) || !std::isfinite(y_fused.y) || !std::isfinite(y_fused.z)) {
				throw std::runtime_error("non-finite fused output");
			}
			fusion_ready = true;
		} catch (const std::exception& ex) {
			log_gate_fallback_warning(ex.what());
			fusion_ready = false;
		} catch (...) {
			log_gate_fallback_warning("unknown exception");
			fusion_ready = false;
		}

		const Vector3f selected = fusion_ready ? y_fused : y_ML;
		if (!fusion_ready) {
			y_fused = y_ML;
		}
		log_training_wide_rows_pidpiper(*this, selected, recoveryMode);
		return selected;
	}

	if (strategyMode == 4) {
		// mode4: 100-frame LSTM gate-NN + original ML fusion
		recoveryMode = true;
		bool fusion_ready = false;

		try {
			const std::vector<float> roll_gate_input = build_gate_input(*this, y_PID.x, y_ML.x, residual[0], kRollMode4GateMean, kRollMode4GateStd);
			const std::vector<float> pitch_gate_input = build_gate_input(*this, y_PID.y, y_ML.y, residual[1], kPitchMode4GateMean, kPitchMode4GateStd);
			const std::vector<float> yaw_gate_input = build_gate_input(*this, y_PID.z, y_ML.z, residual[2], kYawMode4GateMean, kYawMode4GateStd);

			alpha_gate.x = predict_lstm_gate_alpha(roll_mode4_gate_model(), roll_mode4_gate_buffer(), roll_gate_input);
			alpha_gate.y = predict_lstm_gate_alpha(pitch_mode4_gate_model(), pitch_mode4_gate_buffer(), pitch_gate_input);
			alpha_gate.z = predict_lstm_gate_alpha(yaw_mode4_gate_model(), yaw_mode4_gate_buffer(), yaw_gate_input);

			y_fused.x = y_PID.x + alpha_gate.x * (y_ML.x - y_PID.x);
			y_fused.y = y_PID.y + alpha_gate.y * (y_ML.y - y_PID.y);
			y_fused.z = y_PID.z + alpha_gate.z * (y_ML.z - y_PID.z);

			if (!std::isfinite(y_fused.x) || !std::isfinite(y_fused.y) || !std::isfinite(y_fused.z)) {
				throw std::runtime_error("non-finite LSTM fused output");
			}
			fusion_ready = true;
		} catch (const std::exception& ex) {
			log_gate_fallback_warning(ex.what());
			fusion_ready = false;
		} catch (...) {
			log_gate_fallback_warning("unknown exception");
			fusion_ready = false;
		}

		const Vector3f selected = fusion_ready ? y_fused : y_ML;
		if (!fusion_ready) {
			y_fused = y_ML;
		}
		log_training_wide_rows_pidpiper(*this, selected, recoveryMode);
		return selected;
	}

	// fallback for unexpected modes: behave like pure PID
	recoveryMode = false;
	y_ML = y_PID;
	log_training_wide_rows_pidpiper(*this, y_PID, recoveryMode);
	return y_PID;
}
/*
void PID_Piper::write_to_file(Vector3f a, Vector3f g, Vector3f p, Vector3f m, float gV, float aRP, float aYaw, float pVarH, float pVarV, float vVarX, float vVarY, int nRoll, int nPitch, float aspeed)
{
	O_PID_Piper::write_to_file_piper(a, g, p, m, gV, aRP, aYaw, pVarH, pVarV, vVarX, vVarY, nRoll, nPitch, aspeed, y_PID);
}
*/
