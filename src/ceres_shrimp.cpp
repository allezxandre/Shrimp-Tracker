
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
        class InterDistanceCost {
            public:
                InterDistanceCost(double W, double L) : W(W), L2(L*L) {}
                template <typename T> bool operator()( const T* const X1, const T* const X2,  T* residual) const {
                    T d2 = (X2[0] - X1[0])*(X2[0] - X1[0]) + (X2[1] - X1[1])*(X2[1] - X1[1]);
                    residual[0] = T(W)*(d2 - T(L2))/T(L2);
                    return true;
                }

            protected:
                double W;
                double L2;
        };

        class SmoothnessCost {
            public:
                SmoothnessCost(double W, double L) : W(W), L(L){}
                template <typename T> bool operator()( const T* const X1, const T* const X2, const T* const X3,  T* residual) const {
                    T v12[2] = {X2[0] - X1[0], X2[1] - X1[1]};
                    T v23[2] = {X3[0] - X2[0], X3[1] - X2[1]};
                    T nE[2] = {(v23[0] - v12[0])/T(L), (v23[1] - v12[1])/T(L)};
                    residual[0] = T(W)*nE[0];
                    residual[1] = T(W)*nE[1];
                    return true;
                }

            protected:
                double W;
                double L;
        };

        class RotSmoothnessCost {
            public:
                RotSmoothnessCost(double W, double L) : W(W), L2(L*L){}
                template <typename T> bool operator()( const T* const X1, const T* const X2, 
                        const T* const X3, const T* const X4,  T* residual) const {
                    T v12[2] = {X2[0] - X1[0], X2[1] - X1[1]};
                    T v23[2] = {X3[0] - X2[0], X3[1] - X2[1]};
                    T v34[2] = {X4[0] - X3[0], X4[1] - X3[1]};
                    T z123 = v12[0]*v23[1] - v12[1]*v23[0];
                    T z234 = v23[0]*v34[1] - v23[1]*v34[0];
                    T E = (z234/L2 - z123/L2);
                    residual[0] = T(W)*E;
                    return true;
                }

            protected:
                double W;
                double L2;
        };


        class ImageGradientCost  : public ceres::SizedCostFunction<1, 2>  { 

            public:
                ImageGradientCost(double W, unsigned int index,const cv::Mat_<uint8_t> & I, const cv::Mat_<int16_t> & gx, const cv::Mat_<int16_t> & gy) : W(W), index(index), I(I), gx(gx), gy(gy) {}
                virtual ~ImageGradientCost() {}
                virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
                    const double x = parameters[0][0];
                    const double y = parameters[0][1];
                    // Compute the Jacobian if asked for.
                    if (jacobians != NULL && jacobians[0] != NULL) {
                        jacobians[0][0] = 0;
                        jacobians[0][1] = 0;
                    }
                    // printf("%d %.2f %.2f -> ",int(index),x,y);
                    if ((x < 0) || (x>=I.cols)) {
                        residuals[0]= W*1.0;
                    } else if ((y < 0) || (y>=I.rows)) { 
                        residuals[0]= W*1.0;
                    } else {
                        residuals[0] = W*I(int(round(y)),int(round(x)))/255.;
                        // Compute the Jacobian if asked for.
                        if (jacobians != NULL && jacobians[0] != NULL) {
                            jacobians[0][0] = W*gx(int(round(y)),int(round(x)))/255.;
                            jacobians[0][1] = W*gy(int(round(y)),int(round(x)))/255.;
                        }
                    }
                    // printf("%.2f",residuals[0]);
                    // if (jacobians != NULL) {
                    //     printf(" %.2f %.2f",jacobians[0][0],jacobians[0][1]);
                    // }
                    // printf("\n");

                    return true;
                }
            protected:
                double W;
                unsigned int index;
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
            X = new double[2*N];
            L = (I.cols-16.)/(N-1);
            for (size_t i=0;i<N;i++) {
                X[2*i+0] = 8 + i*L;
                X[2*i+1] = I.rows/2;
            }
        }

        ~ShrimpOptimizer() {
            delete [] X;
        }

        typedef enum {ShrimpCurvUp,ShrimpCurvDown,ShrimpStraight} ShrimpType;
        void initialise(ShrimpType type) {
            switch (type) {
                case ShrimpStraight:
                    {
                        for (size_t i=0;i<N;i++) {
                            X[2*i+0] = 8 + i*L;
                            X[2*i+1] = I.rows/2;
                        }
                        break;
                    }
                case ShrimpCurvUp:
                    {
                        float cx = I.cols/2.; float cy = 0;
                        for (size_t i=0;i<N;i++) {
                            X[2*i+0] = cx + (I.cols/2-8)*cos((i+1)*M_PI/(N-1));
                            X[2*i+1] = cy + (I.rows/2)*sin((i+1)*M_PI/(N-1));
                        }
                        break;
                    }
                case ShrimpCurvDown:
                    {
                        float cx = I.cols/2.; float cy = I.rows;
                        for (size_t i=0;i<N;i++) {
                            X[2*i+0] = cx + (I.cols/2-8)*cos(M_PI+(i+1)*M_PI/(N-1));
                            X[2*i+1] = cy + (I.rows/2)*sin(M_PI+(i+1)*M_PI/(N-1));
                        }
                        break;
                    }
                default:
                    break;
            }
        }

        void display() const {
            cv::Mat Ic;
            float ratio = 20;
            cv::resize(I,Ic,cv::Size(),ratio,ratio,cv::INTER_NEAREST);
            for (size_t i=0;i<N;i++) {
                // int x = int(X[2*i+0]);
                // int y = int(X[2*i+1]);
                // float dx = grad_x(y,x);
                // float dy = grad_y(y,x);
                // cv::line(Ic,cv::Point(int(X[2*i+0]*ratio),int(X[2*i+1]*ratio)),
                //         cv::Point(int((X[2*i+0]+dx)*ratio),int((X[2*i+1]+dy)*ratio)), cv::Scalar(255,0,0), 2);
                cv::circle(Ic, cv::Point(int(X[2*i+0]*ratio),int(X[2*i+1]*ratio)), 5, cv::Scalar(0,0,255), -1);
                if (i>0) {
                    cv::line(Ic, cv::Point(int(X[2*(i-1)+0]*ratio),int(X[2*(i-1)+1]*ratio)), 
                            cv::Point(int(X[2*i+0]*ratio),int(X[2*i+1]*ratio)), cv::Scalar(0,0,255), 2);
                }
            }
            cv::imshow("Ic",Ic);
            cv::waitKey(10);
            //cv::waitKey(0);
        }

        void optimizeShrimp() {
            const int scale = 1;
            const int delta = 0;
            cv::Mat_<uint8_t> Ig;
            cv::cvtColor(I,Ig,cv::COLOR_RGB2GRAY);
            /// Generate grad_x and grad_y
            cv::Mat_<uint8_t> abs_grad_x, abs_grad_y;

            cv::GaussianBlur( Ig, Ig, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

            /// Gradient X
            //cv::Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
            cv::Sobel( Ig, grad_x, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );

            /// Gradient Y
            //cv::Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
            cv::Sobel( Ig, grad_y, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

            grad_x.convertTo(abs_grad_x,CV_8U, 1.0,128);
            grad_y.convertTo(abs_grad_y,CV_8U, 0.5,128);

            // cv::imshow("Ig",Ig);
            // cv::imshow("grad_x",abs_grad_x);
            // cv::imshow("grad_y",abs_grad_y);

            Problem problem;

            // Configure the loss function.
            LossFunction* loss = NULL;
            if (FLAGS_robust_threshold>1e-3) {
                loss = new CauchyLoss(FLAGS_robust_threshold);
            }

            // Add the residuals.
            for (size_t i=0;i<N;i++) {
                CostFunction *cost = new ImageGradientCost(1.0,i,Ig, grad_x,grad_y);
                problem.AddResidualBlock(cost, loss, X+2*i);
                if (i>0) {
                    cost = new AutoDiffCostFunction<InterDistanceCost, 1, 2, 2>(
                            new InterDistanceCost(0.5,L));
                    problem.AddResidualBlock(cost, loss, X+2*(i-1),X+2*i);
                }
                if (i > 1) {
                    cost = new AutoDiffCostFunction<SmoothnessCost, 2, 2, 2, 2>(
                            new SmoothnessCost(0.1,L));
                    problem.AddResidualBlock(cost, loss, X+2*(i-2),X+2*(i-1),X+2*i);
                    cost = new AutoDiffCostFunction<InterDistanceCost, 1, 2, 2>(
                            new InterDistanceCost(0.3,L*sqrt(2*(1 - cos(11*M_PI/12)))));
                    problem.AddResidualBlock(cost, loss, X+2*(i-2),X+2*i);
                }
                if (i > 2) {
                    cost = new AutoDiffCostFunction<RotSmoothnessCost, 1, 2, 2, 2, 2>(
                            new RotSmoothnessCost(0.8,L));
                    problem.AddResidualBlock(cost, loss, X+2*(i-3),X+2*(i-2),X+2*(i-1),X+2*i);
                }
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

            // options.minimizer_progress_to_stdout = true;
            // options.minimizer_type = ceres::LINE_SEARCH;
            options.eta = 1e-4;
            options.max_num_iterations = 500;
            options.gradient_tolerance = 1e-6;
            options.function_tolerance = 1e-6;
            // options.linear_solver_type = ceres::DENSE_QR;
            //options.update_state_every_iteration = true; 
            //options.callbacks.push_back(&display_cb);
            double Xbest[2*N], bestCost=0;
            Solver::Summary summary;
            initialise(ShrimpStraight);
            //display();
            //cv::waitKey(0);
            Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << "\n";
            std::copy(X,X+2*N,Xbest);
            bestCost = summary.final_cost;
            //display();
            //cv::waitKey(0);
            initialise(ShrimpCurvUp);
            //display();
            //cv::waitKey(0);
            Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << "\n";
            if (summary.final_cost < bestCost) {
                std::copy(X,X+2*N,Xbest);
                bestCost = summary.final_cost;
            }
            //display();
            //cv::waitKey(0);
            initialise(ShrimpCurvDown);
            //display();
            //cv::waitKey(0);
            Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << "\n";
            if (summary.final_cost < bestCost) {
                std::copy(X,X+2*N,Xbest);
                bestCost = summary.final_cost;
            }
            std::copy(Xbest,Xbest+2*N,X);
            display();
            cv::waitKey(200);

        }
    protected:
        cv::Mat I;
        cv::Mat_<float> grad_x, grad_y;
        double * X;
        double L;
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
        ShrimpOptimizer so(argv[i],12);
        so.optimizeShrimp();
    }
    return 0;
}
