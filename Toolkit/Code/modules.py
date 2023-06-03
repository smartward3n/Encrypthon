import numpy as np
from sklearn.feature_extraction import image


def global_entropy (img):
    
    marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    
    return (entropy)


def local_entropy (img, patch_size: tuple, n_patches: int):
    
    patches = image.extract_patches_2d (image= img, patch_size= patch_size, max_patches= n_patches)
    l_e = global_entropy (patches)
    
    return (l_e)


def H_correlation (img, N: int):
    
    end = img.shape[0]
    x_1 = img [:, 0 : (end - 1)]
    y_1 = img [:, 1 : end]
    
    randIndex_1 = np.random.permutation (np.size (x_1))
    randIndex_1 = randIndex_1 [0 : N]

    temp_x = np.ravel(x_1)
    temp_y = np.ravel(y_1)

    x = [temp_x[i] for i in randIndex_1]
    y = [temp_y[i] for i in randIndex_1]

    E_x = np.multiply((1 / N), np.sum(x))
    E_y = np.multiply((1 / N), np.sum(y))

    D_x = np.multiply((1 / N), np.sum(np.power((np.subtract(x, E_x)), 2)))
    D_y = np.multiply((1 / N), np.sum(np.power((np.subtract(y, E_y)), 2)))

    CO_xy = np.multiply((1 / N), np.sum(np.multiply(np.subtract(x, E_x), np.subtract(y, E_y))))

    H_xy = np.divide(CO_xy, np.sqrt(np.multiply(D_x, D_y)))
    
    return (H_xy)


def V_correlation (img, N: int):
    
    end = img.shape[0]
    x_1 = img [0 : (end - 1), :]
    y_1 = img [1 : end, :]
    
    randIndex_1 = np.random.permutation (np.size (x_1))
    randIndex_1 = randIndex_1 [0 : N]

    temp_x = np.ravel(x_1)
    temp_y = np.ravel(y_1)

    x = [temp_x[i] for i in randIndex_1]
    y = [temp_y[i] for i in randIndex_1]

    E_x = np.multiply((1 / N), np.sum(x))
    E_y = np.multiply((1 / N), np.sum(y))

    D_x = np.multiply((1 / N), np.sum(np.power((np.subtract(x, E_x)), 2)))
    D_y = np.multiply((1 / N), np.sum(np.power((np.subtract(y, E_y)), 2)))

    CO_xy = np.multiply((1 / N), np.sum(np.multiply(np.subtract(x, E_x), np.subtract(y, E_y))))

    V_xy = np.divide(CO_xy, np.sqrt(np.multiply(D_x, D_y)))
    
    return (V_xy)


def D_correlation (img, N: int):
    
    end = img.shape[0]
    x_1 = img [0 : (end - 1), 0 : (end - 1)]
    y_1 = img [1 : end, 1 : end]
    
    randIndex_1 = np.random.permutation (np.size (x_1))
    randIndex_1 = randIndex_1 [0 : N]

    temp_x = np.ravel(x_1)
    temp_y = np.ravel(y_1)

    x = [temp_x[i] for i in randIndex_1]
    y = [temp_y[i] for i in randIndex_1]

    E_x = np.multiply((1 / N), np.sum(x))
    E_y = np.multiply((1 / N), np.sum(y))

    D_x = np.multiply((1 / N), np.sum(np.power((np.subtract(x, E_x)), 2)))
    D_y = np.multiply((1 / N), np.sum(np.power((np.subtract(y, E_y)), 2)))

    CO_xy = np.multiply((1 / N), np.sum(np.multiply(np.subtract(x, E_x), np.subtract(y, E_y))))

    D_xy = np.divide(CO_xy, np.sqrt(np.multiply(D_x, D_y)))
    
    return (D_xy)


def NPCR_and_UACI (cipherImg_1, cipherImg_2, rgb: bool):
    
    if (cipherImg_1.shape != cipherImg_2.shape):
       
        raise Exception ("The input images are not in the same dimension.")
   
    else:
        
        cipherImg_1 = cipherImg_1.astype(np.double)
        cipherImg_2 = cipherImg_2.astype(np.double)
       
        if (rgb == True):
           
            h_1, w_1, c_1 = cipherImg_1.shape
            den = (h_1 * w_1 * c_1)
        
        else:
           
            h_1, w_1 = cipherImg_1.shape
            den = (h_1 * w_1)
           

    NPCR = (np.divide(np.sum (cipherImg_1 != cipherImg_2), den) * 100)
    
    UACI = (np.divide(np.divide(np.sum(np.abs(np.subtract(cipherImg_1, cipherImg_2))), 255), den) * 100)
    
    return (NPCR, UACI)







    
    

    
    


