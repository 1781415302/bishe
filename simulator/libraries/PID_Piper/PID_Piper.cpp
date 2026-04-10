#include "PID_Piper.h"
#include "../../ArduCopter/O_PID_Piper.h"
#include <AP_HAL/AP_HAL.h>
#include <fstream>
#include <mutex>
#include <ctime>
#include <vector>

// ML input logging (记录喂给 ML 的输入，供人工离线判断是否存在攻击)
static std::ofstream ml_input_ofs;
static std::once_flag ml_input_init_flag;
static std::mutex ml_input_mutex;
static const char *ml_input_filename = "/pid-piper/simulator/Data_Piper_ML_Inputs.csv";

// Training wide-table logging (一次仿真直接产出可训练宽表)
static std::ofstream training_wide_ofs;
static std::once_flag training_wide_init_flag;
static std::mutex training_wide_mutex;
static const char *training_wide_filename = "/pid-piper/simulator/Data_Piper_Training_Wide.csv";

static double pidpiper_now_seconds()
{
	return static_cast<double>(AP_HAL::micros64()) * 1.0e-6;
}

static void init_ml_input_file_pidpiper()
{
	std::lock_guard<std::mutex> guard(ml_input_mutex);
	if (ml_input_ofs.is_open()) return;
	ml_input_ofs.open(ml_input_filename, std::ios::out | std::ios::app);
	if (!ml_input_ofs.is_open()) return;
	bool empty = true;
	{
		std::ifstream ifs(ml_input_filename);
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
	training_wide_ofs.open(training_wide_filename, std::ios::out | std::ios::app);
	if (!training_wide_ofs.is_open()) return;
	bool empty = true;
	{
		std::ifstream ifs(training_wide_filename);
		if (ifs.good()) {
			ifs.seekg(0, std::ios::end);
			auto pos = ifs.tellg();
			if (pos > 0) empty = false;
		}
	}
	if (empty) {
		training_wide_ofs << "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,pos_x,pos_y,pos_z,gpsVel,ahrsRP,ahrsYaw,posVarH,posVarV,velVarX,velVarY,navRoll,navPitch,navYaw,angle_type,y_pid,y_ml,residual,attack_label,recovery_mode,y_selected\n";
		training_wide_ofs.flush();
	}
}

static void log_training_wide_row_pidpiper(const PID_Piper &state, const char *angle_name,
		float y_pid, float y_ml, float residual_value, float y_selected, bool recovery_mode)
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
		<< state.attackLabel << "," << (recovery_mode ? 1 : 0) << "," << y_selected << "\n";
	training_wide_ofs.flush();
}

static void log_training_wide_rows_pidpiper(const PID_Piper &state, bool recovery_mode)
{
	const bool use_ml = recovery_mode;
	const Vector3f &selected = use_ml ? state.y_ML : state.y_PID;
	log_training_wide_row_pidpiper(state, "roll", state.y_PID.x, state.y_ML.x, state.residual[0], selected.x, recovery_mode);
	log_training_wide_row_pidpiper(state, "pitch", state.y_PID.y, state.y_ML.y, state.residual[1], selected.y, recovery_mode);
	log_training_wide_row_pidpiper(state, "yaw", state.y_PID.z, state.y_ML.z, state.residual[2], selected.z, recovery_mode);
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
		acc.x = _accX;
		acc.y = _accY;
		acc.z = _accZ;
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
	acc.z = _accZ;
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
	log_ml_input_vec(ml_input, "roll");
	y_ML.x = _lstm.getRollAngle(acc, gyro, pos, mag, gpsVel, ahrsRP, ahrsYaw, posVarH, posVarV, velVarX, velVarY, navRoll, navPitch, airspeed);
	log_ml_input_vec(ml_input, "pitch");
	y_ML.y = _lstm.getPitchAngle(acc, gyro, pos, mag, gpsVel, ahrsRP, ahrsYaw, posVarH, posVarV, velVarX, velVarY, navRoll, navPitch, airspeed);
	log_ml_input_vec(ml_input, "yaw");
	y_ML.z = _lstm.getYawAngle(acc, gyro, pos, mag, gpsVel, ahrsRP, ahrsYaw, posVarH, posVarV, velVarX, velVarY, navRoll, navPitch, airspeed);

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

Vector3f PID_Piper::recoveryMonitor()
{
	y_ML = getEulerAngle();
	/*
	y_PID.x = angle_pid.x;
	y_PID.y = angle_pid.y;
	y_PID.z = angle_pid.z;
	*/

	//difference between ML and PID
	residual[0] = abs(y_ML.x - y_PID.x);
	residual[1] = abs(y_ML.y - y_PID.y);
	residual[2] = abs(y_ML.z - y_PID.z);

	//check cusum
	cusum(residual);

	// check if attack subsides before selecting controller output
	if(recoveryMode)
	{
		recoveryMode = checkSwitchControl();
	}

	log_training_wide_rows_pidpiper(*this, recoveryMode);
	if(recoveryMode)
	{
		return y_ML;
	}
	return y_PID;
}
/*
void PID_Piper::write_to_file(Vector3f a, Vector3f g, Vector3f p, Vector3f m, float gV, float aRP, float aYaw, float pVarH, float pVarV, float vVarX, float vVarY, int nRoll, int nPitch, float aspeed)
{
	O_PID_Piper::write_to_file_piper(a, g, p, m, gV, aRP, aYaw, pVarH, pVarV, vVarX, vVarY, nRoll, nPitch, aspeed, y_PID);
}
*/







