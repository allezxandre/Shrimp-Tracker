
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cstdio>
#include <vector>

#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

DEFINE_double(robust_threshold, 0.0, "Robust loss parameter. Set to 0 for "
        "normal squared error (no robustification).");
DEFINE_string(trust_region_strategy, "levenberg_marquardt",
              "Options are: levenberg_marquardt, dogleg.");
DEFINE_string(dogleg, "traditional_dogleg", "Options are: traditional_dogleg,"
              "subspace_dogleg.");

DEFINE_bool(inner_iterations, false, "Use inner iterations to non-linearly "
            "refine each successful trust region step.");

DEFINE_string(blocks_for_inner_iterations, "automatic", "Options are: "
            "automatic, cameras, points, cameras,points, points,cameras");

DEFINE_string(linear_solver, "sparse_schur", "Options are: "
              "sparse_schur, dense_schur, iterative_schur, sparse_normal_cholesky, "
              "dense_qr, dense_normal_cholesky and cgnr.");
DEFINE_bool(explicit_schur_complement, false, "If using ITERATIVE_SCHUR "
            "then explicitly compute the Schur complement.");
DEFINE_string(preconditioner, "jacobi", "Options are: "
              "identity, jacobi, schur_jacobi, cluster_jacobi, "
              "cluster_tridiagonal.");

class ShrimpOptimizer {
    protected:

        class TrivialEllipsePoint {
            public:
                TrivialEllipsePoint(double theta, double a, double b) : actheta(a*cos(theta)), bstheta(b*sin(theta)) {}
                template <typename T> bool operator()( const T* const X, T* P) const {
                    P[0] = X[0] + T(actheta);
                    P[1] = X[1] + T(bstheta);
                    return true;
                }

            protected:
                double actheta,bstheta;
        };


        class EllipsePoint {
            public:
                EllipsePoint(double theta) : ctheta(cos(theta)), stheta(sin(theta)) {}
                template <typename T> bool operator()( const T* const X, T* P) const {
                    P[0] = X[0] + X[3]*T(ctheta)*cos(X[2]) - X[4]*T(stheta)*sin(X[2]);
                    P[1] = X[1] + X[3]*T(ctheta)*sin(X[2]) + X[4]*T(stheta)*cos(X[2]);
                    return true;
                }

            protected:
                double ctheta,stheta;
        };


        class TrivialImageGradientCost  : public ceres::SizedCostFunction<1, 2>  { 

            public:
                TrivialImageGradientCost(unsigned int index,double theta,double a, double b,
                        const cv::Mat_<uint8_t> & I, const cv::Mat_<int16_t> & gx, const cv::Mat_<int16_t> & gy) :
                    index(index),adEllipse(new TrivialEllipsePoint(theta,a,b)), I(I), gx(gx), gy(gy) {}
                virtual ~TrivialImageGradientCost() {}
                virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
                    double P[2];
                    double J1[2], J2[2];
                    double *J[2] = {J1,J2};
                    // Compute the Jacobian if asked for.
                    if (jacobians != NULL && jacobians[0] != NULL) {
                        adEllipse.Evaluate(parameters,P,J);
                        for (int i=0;i<2;i++) jacobians[0][i] = 0;
                    } else {
                        adEllipse.Evaluate(parameters,P,NULL);
                    }
                    // printf("%d %.2f %.2f -> ",int(index),P[0],P[1]);
                    if ((P[0] < 0) || (P[0]>=I.cols)) {
                        residuals[0]= 0.0;
                    } else if ((P[1] < 0) || (P[1]>=I.rows)) { 
                        residuals[0]= 0.0;
                    } else {
                        residuals[0] = I(int(round(P[1])),int(round(P[0])))/255.;
                        // Compute the Jacobian if asked for.
                        if (jacobians != NULL && jacobians[0] != NULL) {
                            double Ix = gx(int(round(P[1])),int(round(P[0])))/255.;
                            double Iy = gy(int(round(P[1])),int(round(P[0])))/255.;
                            for (int i=0;i<2;i++) {
                                jacobians[0][i] = Ix*J1[i]+Iy*J2[i];
                            }
                        }
                    }
                    // printf("%.3f ",residuals[0]);
                    // printf("\n");
                    return true;
                }
            protected:
                unsigned int index;
                AutoDiffCostFunction<TrivialEllipsePoint, 2, 2> adEllipse;
                const cv::Mat_<uint8_t> I;
                const cv::Mat_<int16_t> gx, gy;
        };

        class ImageGradientCost  : public ceres::SizedCostFunction<1, 5>  { 

            public:
                ImageGradientCost(unsigned int index,double theta,
                        const cv::Mat_<uint8_t> & I, const cv::Mat_<int16_t> & gx, const cv::Mat_<int16_t> & gy) :
                    index(index),adEllipse(new EllipsePoint(theta)), I(I), gx(gx), gy(gy) {}
                virtual ~ImageGradientCost() {}
                virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
                    double P[2];
                    double J1[5], J2[5];
                    double *J[2] = {J1,J2};
                    // Compute the Jacobian if asked for.
                    if (jacobians != NULL && jacobians[0] != NULL) {
                        adEllipse.Evaluate(parameters,P,J);
                        for (int i=0;i<5;i++) jacobians[0][i] = 0;
                    } else {
                        adEllipse.Evaluate(parameters,P,NULL);
                    }
                    // printf("%d %.2f %.2f -> ",int(index),P[0],P[1]);
                    if ((P[0] < 0) || (P[0]>=I.cols)) {
                        residuals[0]= 1.0;
                    } else if ((P[1] < 0) || (P[1]>=I.rows)) { 
                        residuals[0]= 1.0;
                    } else {
                        residuals[0] = I(int(round(P[1])),int(round(P[0])))/255.;
                        // Compute the Jacobian if asked for.
                        if (jacobians != NULL && jacobians[0] != NULL) {
                            double Ix = gx(int(round(P[1])),int(round(P[0])))/255.;
                            double Iy = gy(int(round(P[1])),int(round(P[0])))/255.;
                            for (int i=0;i<5;i++) {
                                jacobians[0][i] = Ix*J1[i]+Iy*J2[i];
                            }
                        }
                    }
                    // printf("%.3f ",residuals[0]);
                    // printf("\n");
                    return true;
                }
            protected:
                unsigned int index;
                AutoDiffCostFunction<EllipsePoint, 2, 5> adEllipse;
                const cv::Mat_<uint8_t> I;
                const cv::Mat_<int16_t> gx, gy;
        };

        class DisplayCallback: public ceres::IterationCallback { 
            protected:
                const ShrimpOptimizer & problem;
            public: 
                DisplayCallback(const ShrimpOptimizer & p) : problem(p) {}

                virtual ceres::CallbackReturnType operator()(const 
                        ceres::IterationSummary& summary) { 
                    problem.display();
                    return ceres::SOLVER_CONTINUE;
                } 
        };

    public:
        ShrimpOptimizer(const char * filename, size_t n) : N(n), display_cb(*this) {
            I = cv::imread(filename);
            cv::resize(I,I,cv::Size(),10.,10.,cv::INTER_LINEAR);
        }

        ~ShrimpOptimizer() {
        }

        void initialize(bool curvatureDown) {
            if (curvatureDown) {
                X[0] = I.cols/2.; X[1] = I.rows; X[2] = 0.0; X[3] = I.cols/2; X[4] = I.rows/2; 
            } else {
                X[0] = I.cols/2.; X[1] = 0; X[2] = 0.0; X[3] = I.cols/2; X[4] = I.rows/2; 
            }
        }

        void display() const {
            cv::Mat Ic;
            float ratio = 2;
            printf("X %f %f %.2f %.2f %.2f\n",X[0],X[1],X[2],X[3],X[4]);
            cv::resize(I,Ic,cv::Size(),ratio,ratio,cv::INTER_NEAREST);
            // Inefficient loop. Could be optimized by reusing results from
            // previous operation. Not critical at this stage
            for (size_t i=0;i<N;i++) {
                double P1d[2], P2d[2];
                double dtheta = 2*M_PI / N;
                EllipsePoint E1(i*dtheta);
                EllipsePoint E2((i+1)*dtheta);
                E1(X,P1d);
                E2(X,P2d);
                cv::Point P1(P1d[0]*ratio,P1d[1]*ratio);
                cv::Point P2(P2d[0]*ratio,P2d[1]*ratio);
                cv::line(Ic, P1, P2, cv::Scalar(0,0,255), 2);
                cv::line(Ic, P1, P2, cv::Scalar(0,0,255), 2);
                cv::circle(Ic,P1,3,cv::Scalar(0,0,255),-1);
                cv::circle(Ic,P2,3,cv::Scalar(0,0,255),-1);
            }
            cv::imshow("Ic",Ic);
            cv::waitKey(100);
            //cv::waitKey(0);
        }

        void optimizeShrimp() {
            const int scale = 1;
            const int delta = 0;
            cv::Mat_<uint8_t> Ig;
            cv::cvtColor(I,Ig,cv::COLOR_RGB2GRAY);
            /// Generate grad_x and grad_y
            cv::Mat_<uint8_t> abs_grad_x, abs_grad_y;

            cv::GaussianBlur( Ig, Ig, cv::Size(11,11), 0, 0, cv::BORDER_DEFAULT );

            /// Gradient X
            //cv::Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
            cv::Sobel( Ig, grad_x, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );

            /// Gradient Y
            //cv::Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
            cv::Sobel( Ig, grad_y, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );


            cv::imshow("Ig",Ig);
            cv::convertScaleAbs(grad_x,abs_grad_x,5.0,0.0);
            cv::convertScaleAbs(grad_y,abs_grad_y,5.0,0.0);
            // grad_x.convertTo(abs_grad_x,CV_8U, 1.0,128);
            // grad_y.convertTo(abs_grad_y,CV_8U, 0.5,128);
            cv::imshow("grad_x",abs_grad_x+abs_grad_y);
            cv::imshow("grad_y",abs_grad_y);

            Problem problem;
            // Configure the loss function.
            LossFunction* loss = NULL;
            if (FLAGS_robust_threshold>1e-3) {
                loss = new CauchyLoss(FLAGS_robust_threshold);
            }

            // Add the residuals.
            for (size_t i=0;i<N;i++) {
                CostFunction *cost = new ImageGradientCost(i,(i * 2 * M_PI) / N, Ig, grad_x,grad_y);
                // CostFunction *cost = new TrivialImageGradientCost(i,(i * 2 * M_PI) / N, I.cols/2, I.rows/2,
                //        Ig, grad_x,grad_y);
                problem.AddResidualBlock(cost, loss, X);
            }

            // Build and solve the problem.
            Solver::Options options;
            CHECK(StringToLinearSolverType(FLAGS_linear_solver,
                        &options.linear_solver_type));
            CHECK(StringToPreconditionerType(FLAGS_preconditioner,
                        &options.preconditioner_type));
            CHECK(StringToTrustRegionStrategyType(FLAGS_trust_region_strategy,
                        &options.trust_region_strategy_type));
            CHECK(StringToDoglegType(FLAGS_dogleg, &options.dogleg_type));
            options.use_inner_iterations = FLAGS_inner_iterations;

            options.minimizer_progress_to_stdout = true;
            // options.minimizer_type = ceres::LINE_SEARCH;
            options.eta = 1e-4;
            options.max_num_iterations = 500;
            options.gradient_tolerance = 1e-10;
            options.function_tolerance = 1e-10;
            options.linear_solver_type = ceres::DENSE_QR;
            options.update_state_every_iteration = true; 
            options.callbacks.push_back(&display_cb);
            Solver::Summary summary;

            initialize(false);
            display();
            cv::waitKey(0);
            Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << "\n";
            display();
            cv::waitKey(0);

            initialize(true);
            display();
            cv::waitKey(0);
            Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << "\n";
            display();
            cv::waitKey(0);

        }
    protected:
        cv::Mat I;
        cv::Mat_<float> grad_x, grad_y;
        double X[5];
        size_t N;
        DisplayCallback display_cb;
};


int main(int argc, char * argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    cv::namedWindow("Ic", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_NORMAL);
    // cv::namedWindow("Ig", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_NORMAL);
    // cv::namedWindow("grad_x", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_NORMAL);
    // cv::namedWindow("grad_y", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_NORMAL);
    for (int i=1;i<argc;i++) {
        ShrimpOptimizer so(argv[i],160);
        so.optimizeShrimp();
    }
    return 0;
}
