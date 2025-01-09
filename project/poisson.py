import cv2
import numpy as np
from scipy.sparse import lil_matrix, linalg

def solve_pixel(dst, mask, lap):
    loc = np.nonzero(mask)
    num = loc[0].shape[0]

    # use lil_matrix to ensure efficiency
    A = lil_matrix((num, num), dtype=np.float64)
    b = np.ndarray((num, ), dtype=np.float64)
    hhash = {(x, y): i for i, (x, y) in enumerate(zip(loc[0], loc[1]))}

    dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]

    for i, (x, y) in enumerate(zip(loc[0], loc[1])):
        A[i, i] = -4
        b[i] = lap[x, y]
        for j in range(4):
            p = (x + dx[j], y + dy[j])
            if p in hhash:  # pixel in mask, value to be solved
                A[i, hhash[p]] = 1
            else:           # pixel on border (delta omega), knwon value
                b[i] -= dst[p]
    X = linalg.splu(A.tocsc()).solve(b)
    X = np.clip(X, 0, 255)

    result = np.copy(dst)
    for i, (x, y) in enumerate(zip(loc[0], loc[1])):
        result[x, y] = X[i]

    return result

def seamless_clone(src, dst, mask, center, flag):
    '''
    WARNING: make sure mask/src size is small enough
    to fit in dst at position(center)
    '''
    src_lap = cv2.Laplacian(src.astype(np.float64), ddepth=-1)

    loc = np.nonzero(mask)
    xbegin, xend, ybegin, yend = np.min(loc[0]) - 1, np.max(loc[0]) + 2, np.min(loc[1]) - 1, np.max(loc[1]) + 2   

    cut_mask = mask[xbegin:xend, ybegin:yend]
    cut_src = src[xbegin:xend, ybegin:yend]
    cut_lap = src_lap[xbegin:xend, ybegin:yend]

    final_mask = np.zeros(dst.shape[:2])
    final_src = np.zeros_like(dst)
    final_lap = np.zeros_like(dst, dtype=np.float64)

    xmid = (np.max(loc[0]) - np.min(loc[0])) // 2 + 1
    xbegin, xend = center[0] - xmid, center[0] + (cut_mask.shape[0] - xmid) 
    ymid = (np.max(loc[1]) - np.min(loc[1])) // 2 + 1
    ybegin, yend = center[1] - ymid, center[1] + (cut_mask.shape[1] - ymid) 

    assert (xbegin >= 0 and ybegin >= 0 and xend < dst.shape[0] and yend < dst.shape[1])

    final_mask[xbegin:xend, ybegin:yend] = cut_mask
    final_src[xbegin:xend, ybegin:yend] = cut_src
    final_lap[xbegin:xend, ybegin:yend] = cut_lap

    if flag == 1:
        kernel = [np.array([[0, -1, 1]]), np.array([[1, -1, 0]]), np.array([[0], [-1], [1]]), np.array([[1], [-1], [0]])]
        temp_grad = []
        for j in range(4):
            src_grad = cv2.filter2D(final_src.astype(np.float64), -1, kernel[j])
            dst_grad = cv2.filter2D(dst.astype(np.float64), -1, kernel[j])
            temp_grad.append(np.where(np.abs(src_grad) > np.abs(dst_grad), src_grad, dst_grad))
        final_lap = np.sum(temp_grad, axis=0)

    final = [solve_pixel(a, final_mask, b) for a, b in zip(cv2.split(dst), cv2.split(final_lap))]
    final = cv2.merge(final)

    return final


dst_path = 'assets/test_image/lake_fake.png'
src_path = 'assets/test_image/wukong.jpg'
mask_path = 'assets/test_image/wukong-m.jpg'
center = (200, 900)
scale = 1.2

dst = cv2.imread(dst_path)
src = cv2.resize(cv2.imread(src_path), dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
mask = cv2.resize(cv2.imread(mask_path), dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)[:,:,0]

result = seamless_clone(src, dst, mask, center, 1)

cv2.imwrite('assets/test_image/Poisson-r.png', result)
