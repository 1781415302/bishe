/*
 * This library uses frugally-deep header only library written by Tobias Hermann
 * https://github.com/Dobiasd/frugally-deep
 */
#ifndef LSTM_H
#define LSTM_H

#include <cstdlib>
#include <fstream>
#include <string>
#include <AP_Math/AP_Math.h>
#include <fdeep/fdeep.hpp>

class LSTM
{
	private:
		Vector3f targetAngle;

		static bool fileExists(const std::string &path) {
			std::ifstream ifs(path.c_str());
			return ifs.good();
		}

		static std::string joinPath(const std::string &base, const std::string &leaf) {
			if (base.empty()) {
				return leaf;
			}
			if (base.back() == '/' || base.back() == '\\') {
				return base + leaf;
			}
			return base + "/" + leaf;
		}

		static std::string resolveModelPath(const char *leaf_name) {
			const char *env_model_dir = std::getenv("PID_PIPER_MODEL_DIR");
			if (env_model_dir != nullptr) {
				const std::string candidate = joinPath(std::string(env_model_dir), std::string(leaf_name));
				if (fileExists(candidate)) {
					return candidate;
				}
			}

			const char *env_root = std::getenv("PID_PIPER_ROOT");
			if (env_root != nullptr) {
				const std::string candidate = joinPath(joinPath(std::string(env_root), "simulator/libraries/PID_Piper/models"), std::string(leaf_name));
				if (fileExists(candidate)) {
					return candidate;
				}
			}

			const std::string candidates[] = {
				joinPath("/pid-piper/simulator/libraries/PID_Piper/models", std::string(leaf_name)),
				joinPath("../libraries/PID_Piper/models", std::string(leaf_name)),
				joinPath("libraries/PID_Piper/models", std::string(leaf_name)),
				joinPath("simulator/libraries/PID_Piper/models", std::string(leaf_name)),
			};
			for (const std::string &candidate : candidates) {
				if (fileExists(candidate)) {
					return candidate;
				}
			}
			return joinPath("/pid-piper/simulator/libraries/PID_Piper/models", std::string(leaf_name));
		}

		fdeep::model loadRollModel() {
			return fdeep::load_model(resolveModelPath("roll-nn.json"));
		}
		fdeep::model loadPitchModel() {
			return fdeep::load_model(resolveModelPath("pitch-nn.json"));
		}
		fdeep::model loadYawModel() {
			return fdeep::load_model(resolveModelPath("yaw-nn.json"));
		}

	public:
		const fdeep::model rollModel = loadRollModel();
		const fdeep::model pitchModel = loadPitchModel();
		const fdeep::model yawModel = loadYawModel();

		float getRollAngle(Vector3f acc, Vector3f gyro, Vector3f pos, Vector3f mag,
				float gpsVel, float ahrsRP, float ahrsYaw, float posVarH, float posVarV,
				float velVarX, float velVarY, int navRoll, int navPitch, float airspeed);

		float getPitchAngle(Vector3f acc, Vector3f gyro, Vector3f pos, Vector3f mag,
				float gpsVel, float ahrsRP, float ahrsYaw, float posVarH, float posVarV,
				float velVarX, float velVarY, int navRoll, int navPitch, float airspeed);

		float getYawAngle(Vector3f acc, Vector3f gyro, Vector3f pos, Vector3f mag,
				float gpsVel, float ahrsRP, float ahrsYaw, float posVarH, float posVarV,
				float velVarX, float velVarY, int navRoll, int navPitch, float airspeed);
};
#endif
