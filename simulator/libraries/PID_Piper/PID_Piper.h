#ifndef PID_PIPER_H
#define PID_PIPER_H

#include <AP_Math/AP_Math.h>
#include <cstdint>
#include "LSTM.h"

class PID_Piper
{
	/*float posX, posY, posZ;
	float accX, accY, accZ;
	float gyroX, gyroY, gyroZ;
	float magX, magY, magZ;
	//float lat, lon, alt;
	float gpsVel;
	float ahrsRP, ahrsYaw;
	float posVarH, posVarV;
	float velVarX, velVarY, compassVar;
	int navRoll, navPitch;
	float airspeed;*/

	public:
		Vector3f acc;
		Vector3f pos;
		Vector3f gyro;
		Vector3f mag;
		Vector3f posControl;
		Vector3f eulerSpeed;
		float gpsVel;
		float ahrsRP, ahrsYaw;
		float posVarH, posVarV;
		float velVarX, velVarY;
		int navRoll, navPitch, navYaw;
		int attackLabel = 0;
		float airspeed;

		Vector3f y_ML;
		Vector3f y_PID;
		Vector3f y_fused;
		Vector3f alpha_gate;
		uint8_t strategyMode = 0;
		double delta[3] = {0,0,0};
		double threshold[3] = {18, 18.09, 18.6};
		double b[3] = {0,0,0};
		double residual[3];
		bool recoveryMode = false;

		void getPosControlXY(float _accX, float _accY, float _accZ,
									float _gyroX, float _gyroY, float _gyroZ,
									float _posX, float _posY, float _posZ,
									float _velX, float _ahrsRP, float _ahrsYaw,
									float _errorX, float _errorY, float _velErrorX, float _velErrorY,
									int _navRoll, int _navPitch, int _navYaw, int _attackLabel);
		void getPosControlZ(float _accZ);

		void getMagnetometerData(float _x, float _y, float _z);

		void getAirSpeed(float _airspeed);

		Vector3f getEulerAngle();
		Vector3f getEulerAngleUser();
		void cusum(double error[3]);
		bool checkSwitchControl();
		void setStrategyMode(uint8_t _mode);

		Vector3f recoveryMonitor();

		LSTM _lstm;

		void write_to_file(Vector3f, Vector3f, Vector3f, Vector3f, float, float, float, float, float, float, float, int, int, float);
};
#endif