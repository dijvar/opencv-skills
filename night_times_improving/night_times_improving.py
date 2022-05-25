'''
https://learnopencv.com/improving-illumination-in-night-time-images/

'''


import cv2
import numpy as np
from guidedfilter import guided_filter




def get_illumination_channel(I, w):
    '''
    Adım 1: Bright and Dark kanalını Alma Öncesi

    İlk adım, parlak ve karanlık kanal önceliklerini tahmin etmektir. Yerel bir yamada sırasıyla maksimum ve minimum 
    piksel yoğunluğunu temsil ederler. Bu prosedür, tüm kanalların maksimum veya minimum değerini bulmamıza yardımcı 
    olan kayan bir evrişimli pencere olarak hayal edilebilir.

    Önce cv2 ve NumPy'yi içe aktarıyoruz ve aydınlatma kanalını almak için işlevi yazıyoruz. Görüntü boyutları M ve N 
    değişkenlerinde saklanır. Boyutlarının aynı kalmasını sağlamak için görüntülere çekirdek boyutunun yarısı kadar 
    dolgu uygulanır. Karanlık kanal, o bloktaki en düşük piksel değerini elde etmek için np.min kullanılarak elde edilir. 
    Benzer şekilde, o bloktaki en yüksek piksel değerini elde etmek için np.max kullanılarak parlak kanal elde edilir. 
    Sonraki adımlar için karanlık kanalın ve parlak kanalın değerine ihtiyacımız olacak.
    '''
    M, N, _ = I.shape
    # kanallar için padding(dolgu)
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])

    return darkch, brightch


def get_atmosphere(I, brightch, p=0.1):
    '''
    2. Adım: Küresel Atmosfer Aydınlatmasının Hesaplanması
    İlk yüzde on yoğunlukların ortalaması alınarak yukarıda elde edilen parlak kanal kullanılarak hesaplanır. Küçük bir 
    anomalinin onu çok fazla etkilememesini sağlamak için değerlerin yüzde onu alınır.

    Bunu kod aracılığıyla gerçekleştirmek için görüntü dizisi yeniden şekillendirilir, düzleştirilir ve maksimum yoğunluğa 
    göre sıralanır. Dizi, piksellerin yalnızca ilk yüzde onu içerecek şekilde dilimlenir ve ardından bunların ortalaması 
    alınır.
    '''
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3) # görüntü dizisini yeniden şekillendirme
    flatbright = brightch.ravel() # flattening görüntü dizisi

    searchidx = (-flatbright).argsort()[:int(M*N*p)] # sıralama ve dilimleme
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A


def get_initial_transmission(A, brightch):
    '''
    Adım 3: İlk İletim Haritasını Bulma

    İletim haritası, ışığın dağılmayan ve kameraya ulaşan kısmını tanımlar. 
    
    Kodda, ilk iletim haritası formül kullanılarak hesaplanır ve daha sonra normalleştirilmiş ilk iletim haritasının 
    hesaplanması için kullanılır.
    '''
    A_c = np.max(A)
    init_t = (brightch-A_c)/(1.-A_c) # ilk iletim haritasını bulma
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # normalleştirilmiş ilk iletim haritası


def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    '''
    Adım 4: Düzeltilmiş İletim Haritasını Tahmin Etmek İçin Karanlık Kanalı Kullanma

    Önceki karanlık kanaldan bir iletim haritası da hesaplanır ve öncelikler arasındaki fark hesaplanır. Bu hesaplama, 
    önceden parlak kanaldan elde edilen potansiyel olarak hatalı iletim tahminlerini düzeltmek için yapılır.

    Fark kanalı alfanın ayarlı değerinden (deneysel bir deney tarafından 0,4 olarak belirlenir) daha düşük olan herhangi 
    bir x pikseli, derinliğini güvenilmez kılan karanlık bir nesnededir. Bu, piksel  x iletimini güvenilmez hale getirir. 
    Bu nedenle, güvenilir olmayan iletim, iletim haritalarının ürünü alınarak düzeltilebilir.
    '''
    im3 = np.empty(I.shape, I.dtype);
    for ind in range(0, 3):
        im3[:, :, ind] = I[:, :, ind] / A[ind] # piksel değerlerini atmosferik ışıkla böl
    dark_c, _ = get_illumination_channel(im3, w) # karanlık kanal iletim haritası
    dark_t = 1 - omega*dark_c # düzeltilmiş karanlık iletim haritası
    corrected_t = init_t # ilk iletim haritası ile düzeltilmiş iletim haritasının başlatılması
    diffch = brightch - darkch # iletim haritaları arasındaki fark

    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if(diffch[i, j] < alpha):
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

    return np.abs(corrected_t)

'''
Adım 5: Kılavuzlu Filtre kullanarak İletim Haritasını Düzgünleştirme

guidedfilter.py

Kılavuzlu görüntü filtreleme, diğer filtreleme işlemleri gibi bir komşuluk işlemidir, ancak çıkış pikselinin değerini 
hesaplarken kılavuz görüntüde karşılık gelen uzamsal komşuluktaki bir bölgenin istatistiklerini hesaba katar.

Özünde, kenar koruyucu bir yumuşatma filtresidir.
'''


def get_final_image(I, A, refined_t, tmin):
    '''
    Adım 6: Sonuç Görüntünün Hesaplanması

    Geliştirilmiş görüntüyü elde etmek için bir iletim haritası ve atmosferik ışık değeri gerekliydi. Artık gerekli 
    değerlere sahip olduğumuza göre, sonucu elde etmek için ilk denklem uygulanabilir.

    İlk olarak, orijinal görüntüdeki ve dönüşüm haritasındaki kanal sayısının aynı olmasını sağlamak için gri tonlamalı 
    rafine dönüşüm haritası gri tonlamalı bir görüntüye dönüştürülür. Daha sonra, çıktı görüntüsü denklem kullanılarak 
    hesaplanır. Bu görüntü daha sonra max-min normalleştirilir ve fonksiyondan döndürülür.
    '''
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3))
    J = (I-A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A

    return (J - np.min(J))/(np.max(J) - np.min(J))

def dehaze(I, tmin, w, alpha, omega, p, eps, reduce=False):
    '''
    Son adım, tüm teknikleri birleştiren ve bir görüntü olarak aktaran bir fonksiyon oluşturmaktır.
    '''
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)

    init_t = get_initial_transmission(A, Ibright) 
    if reduce:
        init_t = reduce_init_t(init_t)
    corrected_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, alpha, omega, w)

    normI = (I - I.min()) / (I.max() - I.min())
    refined_t = guided_filter(normI, corrected_t, w, eps) # kılavuzlu filtre uygulama
    J_refined = get_final_image(I, A, refined_t, tmin)
    
    enhanced = (J_refined*255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15) # Diğer İyileştirmeler
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2) # Diğer İyileştirmeler

    return enhanced

def reduce_init_t(init_t):
    '''
    sınırlamalar

    Görüntülerde lamba gibi açık bir ışık kaynağı veya görüntünün önemli bir bölümünü kaplayan ay gibi doğal bir ışık 
    kaynağı varsa yöntem iyi sonuç vermez. Bu neden bir sorun? Çünkü bu tür ışık kaynakları atmosfer yoğunluğunun 
    değerini yükseltecektir. En parlak piksellerin en üst %10'unu ararken, bu alanların aşırı pozlanmasına neden olacaktır.

    Bunun üstesinden gelmek için, parlak kanal tarafından yapılan ilk iletim haritasını analiz edelim.
    Görev, bu alanların aşırı pozlanmasına neden olan bu yoğun beyaz lekeleri azaltıyor gibi görünüyor. Bu, değerleri 
    255'ten bazı minimum değerlerle sınırlayarak yapılabilir.

    Bunu kodla uygulamak için iletim haritası 0-255 aralığına dönüştürülür. Ardından, orijinal değerlerden yeni bir aralığa 
    noktaları enterpolasyon yapmak için bir arama tablosu kullanılır, bu da yüksek pozlamanın etkisini azaltır.
    '''
    init_t = (init_t*255).astype(np.uint8)
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    init_t = cv2.LUT(init_t, table)
    init_t = init_t.astype(np.float64)/255
    return init_t

im = cv2.imread('dark.png')
orig = im.copy()

tmin = 0.1   # J görüntüsünü oluşturmak için t için minimum değer
w = 15       # önceki görüntülerin doğruluğunu belirleyen pencere boyutu
alpha = 0.4  # iletim düzeltme için threshold
omega = 0.75 # bu, önceki karanlık kanal içindir
p = 0.1      # atmosfer için dikkate alınması gereken yüzde
eps = 1e-3   # J görüntüsü için

I = np.asarray(im, dtype=np.float64) # # Girdiyi bir float array e dönüştürün. 
I = I[:, :, :3] / 255

f_enhanced = dehaze(I, tmin, w, alpha, omega, p, eps)
f_enhanced2 = dehaze(I, tmin, w, alpha, omega, p, eps, True)
cv2.imshow('original', orig)
cv2.imshow('F_enhanced', f_enhanced)
cv2.imshow('F_enhanced2', f_enhanced2)
cv2.waitKey(0)
cv2.destroyAllWindows()
