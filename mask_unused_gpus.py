import os
import subprocess as sp

def mask_unused_gpus(num_gpus=1):
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

    print('masking unused gpus')

    try:
        def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]

        print(memory_free_info)
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        # memory_free_values = [int(x.split()[0])
        #                         for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(
            memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

        if len(available_gpus) < num_gpus:
            raise ValueError('Found only %d usable GPUs in the system' %
                                len(available_gpus))

        aGpus = ','.join(str(x) for x in available_gpus[:num_gpus-1])
        print("using GPU " + aGpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = aGpus

    except Exception as e:
        print('"nvidia-smi" is probably not installed. GPUs are not masked', e)


# if __name__ == '__main__':
#     mask_unused_gpus(2)