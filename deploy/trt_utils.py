import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def get_engine(trt_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    print(f"Reading engine from file {trt_path}")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())  # 反序列化（解码）-> 模型
    
    
def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
    # Calculate start/end binding indices for current context's profile
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile
    
    # Separate input and output binding indices for convenience
    input_binding_idxs = []
    output_binding_idxs = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)

    return input_binding_idxs, output_binding_idxs


def normlize_cuda(channel_fist=True):
    norm = SourceModule(
        """
            __global__ void NormMeanStd_HWC(unsigned char *src, float *dst, float *mean, float *std)
            {
                unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
                dst[3 * index + 0] = (src[3 * index + 0] / 255.0 - mean[0]) / std[0];
                dst[3 * index + 1] = (src[3 * index + 1] / 255.0 - mean[1]) / std[1];
                dst[3 * index + 2] = (src[3 * index + 2] / 255.0 - mean[2]) / std[2];
            }
            __global__ void NormMeanStd_CHW(unsigned char *src, float *dst, int img_h, int img_w, float *mean, float *std)
            {
                int index = blockIdx.x * (blockDim.x * blockDim.y)  + threadIdx.y * blockDim.x  + threadIdx.x;
                if(index < img_h * img_w)
                {
                    dst[index] = (src[3 * index + 0] / 255.0 - mean[0]) / std[0];
                    dst[index + img_h * img_w] = (src[3 * index + 1] / 255.0 - mean[1]) / std[1];
                    dst[index + 2 * img_h * img_w] = (src[3 * index + 2] / 255.0 - mean[2]) / std[2];
                }
            }
        """
    )
    if channel_fist:
        # 输入 h w c 输出 c h w
        return norm.get_function("NormMeanStd_CHW")
    else:
        # 输入 h w c 输出 h w c
        return norm.get_function("NormMeanStd_HWC")
        