TF_INC=/usr/lib/python3.7/site-packages/tensorflow/include
LD_LIBRARY_PATH=/opt/cuda/lib64
TF_LIB=/usr/lib/python3.7/site-packages/tensorflow

all: winograd2x2_imTrans_op.so winograd2x2_imTrans_grad_op.so

forward: winograd2x2_imTrans_op.so

backward: winograd2x2_imTrans_grad_op.so

winograd2x2_imTrans_op.so: winograd2x2_imTrans_op.cc winograd2x2_imTrans_cuda.cu.o
	g++ -std=c++11 -shared -o winograd2x2_imTrans_op.so winograd2x2_imTrans_op.cc winograd2x2_imTrans_cuda.cu.o -I $(TF_INC) -fPIC -L$(LD_LIBRARY_PATH) -lcudart -O2 -L$(TF_LIB) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=1

winograd2x2_imTrans_cuda.cu.o: winograd2x2_imTrans_cuda.cu.cc
	nvcc -std=c++11 -c -o winograd2x2_imTrans_cuda.cu.o winograd2x2_imTrans_cuda.cu.cc -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch sm_52 -L$(TF_LIB) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=1 # --use_fast_math
		
winograd2x2_imTrans_grad_op.so: winograd2x2_imTrans_grad_op.cc winograd2x2_imTrans_grad_cuda.cu.o
	g++ -std=c++11 -shared -o winograd2x2_imTrans_grad_op.so winograd2x2_imTrans_grad_op.cc winograd2x2_imTrans_grad_cuda.cu.o -I $(TF_INC) -fPIC -L$(LD_LIBRARY_PATH) -lcudart -O2 -L$(TF_LIB) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=1

winograd2x2_imTrans_grad_cuda.cu.o: winograd2x2_imTrans_grad_cuda.cu.cc
	nvcc -std=c++11 -c -o winograd2x2_imTrans_grad_cuda.cu.o winograd2x2_imTrans_grad_cuda.cu.cc -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch sm_52 -L$(TF_LIB) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=1 # --use_fast_math
	
clean:
	rm *.o *.so
