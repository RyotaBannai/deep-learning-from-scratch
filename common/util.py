# coding: utf-8
import numpy as np


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[5 : len(y) - 5]


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2 * pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    # ↓p213 畳み込み層の出力サイズに関する計算。
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    """
    np.pad の第2引数 [(0,0),(0,0),(pad,pad),(pad,pad)] は各軸ごとの前後パディング量を指定しています。
    ここでは軸0（N）と軸1（C）にはパディングせず、軸2（高さ）と軸3（幅）に対してそれぞれ前後 pad ピクセルずつ追加します。
    mode="constant" は定数で埋めることを意味し、デフォルトでは 0（ゼロパディング）になります。
    結果として得られる img の形状は (N, C, H + 2pad, W + 2pad) になります。
    目的は畳み込みの前処理で
    """
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # col.shape
    #   軸1: フィルターの適用回数（データ数*畳み込み層の出力サイズのout_h*out_w）
    #   軸2: フィルターの重みshape（channel*FH*FW）

    # col.reshape後.shape
    #   軸1: データ数
    #   軸2: FH
    #   軸3: FW
    #   軸4: out_h
    #   軸5: out_w
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    # img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    # 上記のようにstride-1はバッファー観点で入れている。
    # 実際にはout_h,out_wは//strideでfloorしているため、H+2*pad,W+2*padだけ確保すればout of indexになることはない。
    # 証明: y+(out_h-1)*stride <= (FH-1) + floor((H_p - FH) /stride)* stride <=(FH-1)+(H_p-HF)=H_p-1。つまり最大indexはH_p-1である。
    # ※H_p=H+2*pad を表す。
    # ※y+(out_h-1)*strideが最大indexアクセスになることの具体歴で証明：
    #   y=0,H=8,pad=0,filter_h=3,stride=2,out_h=3（=(H+2*pad-filter_h)//stride+1）の時、y_max=0+3*2=6
    #   slice y:y_max:2=[0,2,4]
    #   最大のindexは4であるが、これは(out_h-1)*stride=2*2=4と一致（また、y_max-stride=6-2=4としても一致）。

    # それでも一部実装が + stride - 1 を付けるのは、保守的なバッファ（境界条件に対する安全側の余裕）で、正しさのために必須だからではありません。また最後に [:, :, pad:H+pad, pad:W+pad] で必ずトリミングするので、余白は捨てられます。
    # とはいえ、実装上はH+2*padで十分であり、想定外のバグが発生した場合にもバッファーで吸収する可能性もあるため、バッドプラクティスと思われる。
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : H + pad, pad : W + pad]
