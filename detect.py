from multiprocessing.pool import ThreadPool


def detect(model, image):
    pool = ThreadPool(processes=10)
    async_result = pool.apply_async(model, (image,)) # tuple of args for foo
    result = async_result.get()
    boxes = result.pandas().xyxy[0]
    boxes = boxes.iloc[:,:4].values
    return boxes