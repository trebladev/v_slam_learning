#include <iostream>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main()
{
	Quaterniond q1;
	Quaterniond q2;

	q1 = Quaterniond(0.35,0.2,0.3,0.1);
	q2 = Quaterniond(-0.5,0.4,-0.1,0.2);

	q1.normalize();
	q2.normalize();

	Isometry3d T1w(q1),T2w(q2);
	T1w.pretranslate(Vector3d(0.3,0.1,0.1));
	T2w.pretranslate(Vector3d(-0.1,0.5,0.3));

	Vector3d Pr1 = Vector3d(0.5,0,0.2);
	Vector3d Pr2;
	Pr2 = T2w*T1w.inverse()*Pr1;
	cout<<"Pr2 = "<<Pr2.transpose()<<endl;
	return 0;
}


