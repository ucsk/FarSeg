import cv2
import numpy as np
import paddlers as pdrs


def run_dynamic(model_dir: str, img_file):
    model = pdrs.tasks.load_model(model_dir)
    result = model.predict(img_file)
    return result


def run_static(model_dir: str, img_file):
    predictor = pdrs.deploy.Predictor(
        model_dir=model_dir,
        use_gpu=True,
        gpu_id=0,
        cpu_thread_num=1,
        use_mkl=True,
        mkl_thread_num=4,
        use_trt=False,
        use_glog=False,
        memory_optimize=True,
        max_trt_batch_size=1,
        trt_precision_mode='float32')
    result = predictor.predict(
        img_file=img_file,
        warmup_iters=0,
        repeats=1)
    return result


if __name__ == '__main__':
    image_path = 'infer_test/P1854_3391_4287_9216_10112.png'
    mask1 = run_dynamic('normal_model', image_path)['label_map']
    mask2 = run_static('inference_model', image_path)['label_map']
    assert np.sum(mask1 - mask2) == 0
    cv2.imwrite('infer_test/dynamic.png', mask1 * 16)
    cv2.imwrite('infer_test/static.png', mask2 * 16)
