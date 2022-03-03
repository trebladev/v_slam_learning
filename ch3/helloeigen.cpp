#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace std;
int main()
{
	//std::cout<< "helloworld" <<std::endl;
	Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Random();
	cout<< matrix_33 << endl;
	cout<< matrix_33.sum() <<endl;
	cout<< matrix_33.trace() <<endl;
	cout<<matrix_33.inverse()<<endl;
	cout<< matrix_33*matrix_33.inverse() <<endl;

	Eigen::Matrix<double ,50,50> matrix_NN;
	matrix_NN = Eigen::MatrixXd::Random(50,50);
	Eigen::Matrix<double,50,1> v_Nd;
	v_Nd = Eigen::MatrixXd::Random(50,1);
	clock_t time_stt = clock();

//	Eigen::Matrix<double,50,1> x = matrix_NN.inverse()*v_Nd;
//	cout<< "time use in normal inverse is"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
//	cout<< x <<endl;	
//	time_stt = clock();
	Eigen::Matrix<double,50,1> x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	cout<< "time use in normal inverse is"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
//	cout<< x <<endl;

	return 0;
}
