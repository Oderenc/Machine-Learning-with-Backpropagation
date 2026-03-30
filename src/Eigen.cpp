#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;

//Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
Matrix<double, 5, 5> testMarix;
//Comes uninitialized

//Vectors(for my purposes) are matrixes with one col
Matrix<double, 5, 1> vector;

//You can also have dynamic matrixes:
MatrixXd a(5, 5);
//Matrix XD is a double, initialized in compile time.

VectorXd b(5);

//Writing to matrices
/*
int main() {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << "Here is the matrix m:\n" << m << std::endl;
  Eigen::VectorXd v(2);
  v(0) = 4;
  v(1) = v(0) - 1;
  std::cout << "Here is the vector v:\n" << v << std::endl;
}
*/

/*

When should one use fixed sizes (e.g. Matrix4f), and when should one prefer dynamic sizes (e.g. MatrixXf)? 
The simple answer is: use fixed sizes for very small sizes where you can, and use dynamic sizes for larger sizes or where you have to. 
For small sizes, especially for sizes smaller than (roughly) 16, using fixed sizes is hugely beneficial to performance, 
as it allows Eigen to avoid dynamic memory allocation and to unroll loops.
*/

//Addition & subtraction
/*
int main() {
  Eigen::Matrix2d a;
  a << 1, 2, 3, 4;
  Eigen::MatrixXd b(2, 2);
  b << 2, 3, 1, 4;
  std::cout << "a + b =\n" << a + b << std::endl;
  std::cout << "a - b =\n" << a - b << std::endl;
  std::cout << "Doing a += b;" << std::endl;
  a += b;
  std::cout << "Now a =\n" << a << std::endl;
  Eigen::Vector3d v(1, 2, 3);
  Eigen::Vector3d w(1, 0, 0);
  std::cout << "-v + w - v =\n" << -v + w - v << std::endl;
}
*/

//Scalar & matrix multiplication
/*
int main() {
  Eigen::Matrix2d a;
  a << 1, 2, 3, 4;
  Eigen::Vector3d v(1, 2, 3);
  std::cout << "a * 2.5 =\n" << a * 2.5 << std::endl;
  std::cout << "0.1 * v =\n" << 0.1 * v << std::endl;
  std::cout << "Doing v *= 2;" << std::endl;
  v *= 2;
  std::cout << "Now v =\n" << v << std::endl;
}
*/

//Vector * Matrix
/*
int main() {
  Eigen::Matrix2d mat;
  mat << 1, 2, 3, 4;
  Eigen::Vector2d u(-1, 1), v(2, 0);
  std::cout << "Here is mat*mat:\n" << mat * mat << std::endl;
  std::cout << "Here is mat*u:\n" << mat * u << std::endl;
  std::cout << "Here is u^T*mat:\n" << u.transpose() * mat << std::endl;
  std::cout << "Here is u^T*v:\n" << u.transpose() * v << std::endl;
  std::cout << "Here is u*v^T:\n" << u * v.transpose() << std::endl;
  std::cout << "Let's multiply mat by itself" << std::endl;
  mat = mat * mat;
  std::cout << "Now mat is mat:\n" << mat << std::endl;
}
*/

//g++ -O3 -I "C:\Users\abuch\Downloads\eigen-5.0.0"