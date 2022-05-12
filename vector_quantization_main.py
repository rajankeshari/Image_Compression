import numpy as np
import cv2
import math
from vector_quantization import *
import matplotlib.pyplot as plt

def get_bandwidth():
    bdwidth = [1,2,3,4,5,6]
    for i in range(100):
        yield bdwidth[i%6]

def decompressed_block(list_block, bs = 4):
    block = np.zeros((bs,bs), dtype = np.uint8)

    H, L = list_block[0], list_block[1]
    ptr = 2

    for i in range(bs):
        for j in range(bs):
            if list_block[ptr] == 1:
                block[i][j] = H
            else:
                block[i][j] = L
            ptr += 1
    
    return block

def decompress_image(img_cmp, bs = 4):
    m, n = img_cmp[0], img_cmp[1]

    img_final = np.zeros((m,n), dtype = np.uint8)
    img_cmp = img_cmp[2:]

    b = bs*bs + 2

    ptr = 0
    row = m//bs
    col = n//bs

    for i in range(row):
        for j in range(col):
            block = decompressed_block(img_cmp[ptr: ptr + (2 + bs*bs)], bs)
            img_final[bs*i:bs*i+bs, bs*j:bs*j+bs] = block
            ptr += (2 + bs*bs)
    return img_final

class Block_Truncation:
    def __init__(self, img, block_size = 4):
        bs = block_size
        m, n = img.shape
        rp = 0
        cp = 0
        if m%bs != 0:
            rp = bs-(m%bs)
        if n%bs != 0:
            cp = bs-(n%bs)
        
        temp = np.zeros((m+rp, n+cp), dtype = np.uint8)
        temp[0:m, 0:n] = img

        m, n = temp.shape
        row = m//bs
        col = n//bs

        compressed = [m, n]
        for i in range(row):
            for j in range(col):
                block = img[bs*i:bs*i+bs, bs*j:bs*j+bs]
                list_block = self.get_compressed_block(block)
                compressed.extend(list_block)
        
        self.compressed = compressed
    
    def calc_HL(self, block):
        m, n = block.shape
        sigma = np.std(block)
        mu = np.mean(block)
        epsilon = 0.0001
        temp = block.copy()
        temp[block >= mu] = 1
        temp[block < mu] = 0

        q = np.sum(temp)

        H = int(round(mu + sigma * math.sqrt((m*n - q)/(q+epsilon))))
        L = int(round(mu - sigma * math.sqrt(q/(m*n - q + epsilon))))

        if H > 255:
            H =255
        if L < 0:
            L = 0

        return H, L, mu

    def get_compressed_block(self, block):
        H, L, mu = self.calc_HL(block)
        out = block.copy()
        out[block >= mu] = 1 
        out[block < mu] = 0

        output = [H, L]
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                output.append(out[i][j])

        return output

def preprocess(frame, bs):
    m, n = frame.shape
    final_m = (m//bs)*bs
    final_n = (n//bs)*bs
    if m%bs != 0:
        final_m += bs
    if n%bs != 0:
        final_n += bs
    frame_new = np.zeros((final_m, final_n), dtype = np.uint8)
    frame_new[0:m, 0:n] = frame

    return frame_new

BLOCK_SIZE = 16
BLUR_ADJUST = (7,7)
FPS = 2


if __name__ == '__main__':
    itr = get_bandwidth()
    cam = cv2.VideoCapture(0)

    if cam.isOpened() == False:
        print('Erorr opening video!!')
    ret, frame1 = cam.read()
    image_size = (frame1.shape[0], frame1.shape[1])

    PSNR_ratio_list=[]
    RMSE_noise_list=[]

    while cam.isOpened():
        flag = None
        img_cmp = None
        out_frame = None
        # bw = next(itr)
        bw = 3
        ret, frame1 = cam.read()
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        frame = preprocess(frame, BLOCK_SIZE)
        if bw < 5:
            img_cmp = Block_Truncation(frame, BLOCK_SIZE).compressed 
            out_frame = decompress_image(img_cmp, BLOCK_SIZE)
            out_frame = cv2.blur(out_frame, BLUR_ADJUST)    #Tweak in this parameter to adjust blur (is always odd - (o,o))

        if bw > 5:
            codebook, img_cmp = vq_compress_image(frame1)
            out_frame = vq_decompress_image(codebook, img_cmp, frame1.shape[0], frame1.shape[1])
            # out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
            out_frame = cv2.blur(out_frame, BLUR_ADJUST)    #Tweak in this parameter to adjust blur (is always odd - (o,o))
        
        SE = 0
        
        m, n, k= frame1.shape
        for i in range(m):
            for j in range(n):
                SE = SE+(int(frame1[i][j][0])-int(out_frame[i][j]))**2
        
        MSE = SE/(m*n)
        PSNR = 10*math.log((255*255/MSE),10)
        RMSE = MSE**(0.5)
        #noise = noise**(0.5)
        #print("Frame noise is : ",PSNR)
        #print(type(noise))
        PSNR_ratio_list.append(PSNR)
        RMSE_noise_list.append(RMSE)
        
        cv2.imshow('Input Video Feed', frame1)
        cv2.imshow('Output Video Feed', out_frame)


        if cv2.waitKey(1) == ord('q'):
            break
    

    print("\n\n\nCompression ratio using Vector Quantization is = 8:1\n\n\n")
    n = len(PSNR_ratio_list)
    #print("Average PSNR noise is : ", sum(PSNR_noise_list)/n)
    frames = [ i for i in range(n) ]
    #print(frames,PSNR_noise_list)
    plt.ylim(20,30)
    plt.xlim(1,50)
    plt.plot(frames, PSNR_ratio_list)
    plt.xlabel('Frame')
    plt.ylabel('PSNR Ratio')   
    plt.title('PSNR ratio graph')
    plt.show()
    
    plt.ylim(10,20)
    plt.xlim(1,50)
    plt.plot(frames, RMSE_noise_list)
    plt.xlabel('Frame')
    plt.ylabel('RMSE Noise')   
    plt.title('RMSE noise graph')
    plt.show()

    cam.release()
    cv2.destroyAllWindows()