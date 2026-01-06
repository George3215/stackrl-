"""
Baseline policies with no learning, using heuristics
"""
#导入标准库，用于时间戳、日志记录、实验时间标记
from datetime import datetime
import time #用于计时、sleep、性能统计
import os #用于路径、环境变量、文件系统操作

import gin #Google 的 Gin-config 配置系统，用于将超参数与代码解耦
import gym #OpenAI Gym 接口，用于定义和交互强化学习环境
import numpy as np #数值计算核心库，在本文件中主要用于矩阵运算、滑窗计算、mask 操作等
from scipy import signal #用于二维相关（correlate2d）、卷积等信号处理操作
from scipy import ndimage #用于图像级操作，如 sobel、minimum_filter
try:
  import cv2 as cv
except :
  cv = None

# from stackrl import envs
from stackrl import agents

def get_inputs(inputs, mask=None):   #从环境返回的 inputs 中提取并归一化用于计算启发式值的数组
       #mask 参数在这里未使用，只是为了接口统一（与其他 heuristic 函数保持签名一致）
  """Extract and normalize the arrays from inputs"""
  gmax = inputs[0][:,:,1].max()  #inputs[0]：通常表示当前场景 / 地形状态（例如高度图 + 目标图），整个场景中，目标高度通道的最大值
  o = inputs[0][:,:,0]/gmax#当前地形高度
  n = inputs[1][:,:,0]/gmax#待放置物体高度
  return o,n

#基于高度的启发式策略函数
def height(inputs, mask=None, **kwargs): #把当前物体 n 放到场景 o 的每一个可能位置后，形成的最大高度
  #mask = None的意思是，如果不传入参数，mask就是None。**kwargs 代表 任意数量的关键字参数，在调用的时候即使传入了没有声明的变量也不会报错
  """Height based heuristic."""
  o,n = get_inputs(inputs, mask) #  o：归一化后的 场景当前高度图 (H, W)    n：归一化后的 待放置物体高度图 (h, w)
  f = np.zeros(np.subtract(o.shape, n.shape)+1) #创建一个二维数组 f，用于存储“物体在每一个合法放置位置时的评价值（代价）”
  #subtract是相减
  n_where = n > 0 #将n>0的变为true，是有效格子，n<0变为false是无效格子
 # 在所有合法放置位置上，计算“放上这个物体之后，整体最高会有多高”，并优先选择“不会把堆叠高度抬得太高”的位置。
  for i in range(f.shape[0]):
    for j in range(f.shape[1]):
      if mask is None or mask[i,j]:
        f[i,j] = np.max(np.where(
          n_where,          # 条件：物体哪些位置有效，是布尔值true/false
          o[i:i+n.shape[0], j:j+n.shape[0]] + n,  # 将物体放到场景对应区域后 叠加高度 # 条件为 True → 用叠加后的高度
          0, #o_sub = o[i:i+h, j:j+w] 是场景中对应位置的高度    # 条件为 False → 空白位置高度设为 0
        ))

  return f


# 计算每个可能放置位置的“局部不平整度”  选择堆叠后高度变化最小的位置，避免把物体放得太不平整 ，位置加权幂次，用于更关注物体中心或边缘
def difference(inputs, mask=None, difference_exponent=2, weights_exponent=2, return_height=False, **kwargs):
#inputs：输入场景和待放物体的高度图        mask：可选的合法放置位置掩码       difference_exponent：高度差的幂次，用于放大高差影响
#weights_exponent：加权幂次，用于强调物体中心或边缘      return_height：是否同时返回局部最大高度      **kwargs：允许传入额外参数给其他函数
  """Difference based heuristic."""
  o,n = get_inputs(inputs)

  #Python 的数组索引是 左闭右开，所以为了取到(H-h, W-w)，就必须+1
  f = np.zeros(np.subtract(o.shape, n.shape)+1)    #f ：存储每个放置位置的**“不平整代价值”**     计算所有合法放置位置的数量
  height = np.zeros_like(f) if return_height else None    # 存储局部最大高度
  #<结果1> if <条件> else <结果2>    如果 <条件> 为 True，就返回 <结果1>   如果 <条件> 为 False，就返回 <结果2>
  
  n_where = n > 0 #只考虑物体实际占据的格子，n_where 是布尔矩阵，标记物体有效的高度格子

  if weights_exponent > 0:   #是否使用中心加权，weights_exponent > 0 → 使用 中心加权
    _wi = (np.arange(n.shape[0], dtype='float') - n.shape[0]/2)**2   #生成物体高度图行索引数组，并计算每行距离物体中心的平方
    _wj = (np.arange(n.shape[1], dtype='float') - n.shape[1]/2)**2   #生成列索引数组，并计算每列距离物体中心的平方
    w = (_wi[:,np.newaxis] + _wj[np.newaxis,:])**(weights_exponent/2)  #把行和列距离平方相加 → 得到每个格子到物体中心的距离平方和
    w = np.where(n_where, w, 0)   # 是布尔矩阵，只标记物体实际占据的格子
    w /= w.sum()   # 确保所有权重之和为 1
  else:
    w = n_where.astype('float')  #不使用中心加权，给每个格子相等权重，.astype('float') → 把布尔值转换成浮点数
    w /= w.sum()

  
  for i in range(f.shape[0]):
    for j in range(f.shape[1]):
      if mask is None or mask[i,j]:        # mask 是可选的布尔矩阵，标记哪些位置是允许放置的，如果 mask 为 None → 所有位置都合法
        h = o[i:i+n.shape[0], j:j+n.shape[0]] + n   # 取出场景 o 中放置物体 n 区域对应的高度片段
        h0 = np.max(np.where(n_where, h, 0))    # h0是放置物体后该区域的高度
        f[i,j] = np.sum(w*np.abs(h0 - h)**difference_exponent) # h0 - h → 每个格子高度与局部最高点的差，使用w权重矩阵进行加权
        #计算“局部不平整度”，也就是衡量把物体放在当前位置 (i,j) 后，堆叠表面有多不平整
        
        if height is not None:   # 记录局部最大高度
          height[i,j] = h0 

  if height is not None:
    f = f, height。 #  f = f, height 不是数学赋值，而是 Python 的元组打包（tuple packing）
    #把两个数组打包成一个元组，让函数一次返回两个结果，不是“同时赋值”，不是“追加”，而是 变量 f 的类型发生了改变。
  return f


def corrcoef(inputs, mask=None, localized=False, **kwargs):  
  # 把待放置物体 n 当作一个“模板”，在场景高度图 o 上滑动，计算每个可能位置处二者的“相关系数”，用来衡量“形状是否匹配”。
  """Correlation coefficient based heuristic."""
  o,n = get_inputs(inputs)

  if not localized:   # 是不是“非局部相关”（localized=False）
    if cv is not None:     # 模板匹配（Template Matching）函数，用于在一张大图中滑动搜索一张小图（模板），并计算每个位置的相似度或差异评分。
      return cv.matchTemplate(o.astype('float32'),n.astype('float32'),cv.TM_CCOEFF_NORMED)#归一化相关系数
      # 把物体高度图 n 当作模板，在场景高度图 o 上滑动，计算每一个可能放置位置的“归一化相关系数”，并把整张评分图直接返回
      
    n_where = np.ones_like(n, dtype='bool')  #np.ones_like(n)：形状和 n 一样     全部是 True
  else:
    n_where = n > 0   #  只关心“物体区域” 

  #手写的归一化互相关（Normalized Cross-Correlation, NCC），都是在计算两个高度图（模板 n 与场景 o 的局部片段）之间的相关性。
  f = np.zeros(np.subtract(o.shape, n.shape)+1)

  n_count = np.count_nonzero(n_where)
  n -= np.sum(np.where(n_where, n, 0))/n_count
  n_var = np.sum(np.where(n_where, n**2, 0))

  if n_var == 0:
    # No need to calculate, everything will be zero
    return f

  for i in range(f.shape[0]):
    for j in range(f.shape[1]):
      if mask is None or mask[i,j]:
        o_ = o[i:i+n.shape[0], j:j+n.shape[0]] - np.sum(np.where(
          n_where,
          o[i:i+n.shape[0], j:j+n.shape[0]],
          0,
        ))/n_count
        o_var = np.sum(np.where(n_where, o_**2, 0))

        if o_var != 0:
          f[i,j] = np.sum(np.where(n_where, n*o_, 0))/np.sqrt(n_var*o_var)

  return f

def gradcorr(inputs, sobel=False, **kwargs):
  """Correlation coefficient based heuristic."""
  # 使用梯度相关系数（Gradient Correlation）来评估物体 n 放置在场景 o 的每个位置的匹配程度。
  # sobel=True 时使用 Sobel 算子计算梯度，否则使用 np.gradient。
  o,n = get_inputs(inputs)  # o: 场景高度图, n: 待放置物体高度图

  uniform = np.ones_like(n)  # uniform 是一个与 n 同尺寸的全 1 数组，用于加权卷积

  if sobel:   # 使用 Sobel 算子分别计算 x 轴和 y 轴梯度
    o_dx, o_dy = ndimage.sobel(o, axis=0), ndimage.sobel(o, axis=1)
    n_dx, n_dy = ndimage.sobel(n, axis=0), ndimage.sobel(n, axis=1)
  else:    # 使用 np.gradient 计算梯度（数值差分）
    o_dx, o_dy = np.gradient(o)
    n_dx, n_dy = np.gradient(n)

  # 计算归一化分母部分（局部梯度平方和的卷积）
  vx = signal.correlate2d(o_dx**2, uniform, mode='valid')*np.sum(n_dx**2)
  vy = signal.correlate2d(o_dy**2, uniform, mode='valid')*np.sum(n_dy**2)

  # 计算 x 方向的梯度相关系数
  fx = signal.correlate2d(o_dx, n_dx, mode='valid')/np.sqrt(np.where(
    vx, vx, 1.
  ))
  
  # 计算 y 方向的梯度相关系数
  fy = signal.correlate2d(o_dy, n_dy, mode='valid')/np.sqrt(np.where(
    vy, vy, 1.
  ))

  return (fx + fy)/2

def correlate(inputs, **kwargs):
  o,n = get_inputs(inputs)
  
  #计算二维互相关（cross-correlation），相当于把 n 当作模板在 o 上滑动匹配，每个位置得到一个匹配分数
  #每个位置得到一个匹配分数
  #mode='valid' → 只计算完整重叠的位置，不超出边界
  return signal.correlate2d(o,n,mode='valid')/n.sum()  

def random(inputs, seed=None, **kwargs): # 创建随机数生成器，如果传入 seed，可以保证每次生成相同的随机数
  """Returns random values in the same shape as the heuristics."""
  rng = np.random.default_rng(seed)
  return rng.random(
    (np.subtract(inputs[0].shape, inputs[1].shape)[:-1]+1)
  )  # 生成随机矩阵并返回，随机矩阵形状为 output_shape，每个元素在 [0, 1) 之间

def goal_overlap(inputs, threshold=0.75, **kwargs):  # 找出物体放置位置与“目标高度区域”的重叠度足够高的位置。
  b = (inputs[0][:,:,0] < inputs[0][:,:,1]).astype('int')  # 场景中还需要填充的目标位置（目标高度比当前高度高）
  n = (inputs[1][:,:,0] > 0).astype('int') # 待放置物体占据的格子
  f = signal.correlate2d(b, n, mode='valid')
  return f >= threshold*f.max()

methods = {
  'random': random,
  'correlate':correlate,
  'height': height,
  'difference': difference,
  'corrcoef':corrcoef,
  # 'gradcorr':gradcorr,
}

@gin.configurable(module='stackrl')
class Baseline(agents.PyGreedy):
  def __init__(
    self, 
    method='random', 
    goal=True,
    minorder=1,
    value=False, 
    unravel=False,
    batched=False,
    batchwise=False,
    **kwargs,
  ):
    if isinstance(method, str):
      if method in methods:
        method = methods[method]
      else:
        raise ValueError(
          "Invalid value {} for argument method. Must be in {}".format(method, methods)
        )
    elif not callable(method):
      raise TypeError(
        "Invalid type {} for argument method.".format(type(method))
      )

    self.model = method
    self.goal = goal
    self.kwargs = kwargs
    self.minorder = minorder    
    self.value = value
    self.unravel = unravel
    self.batched = batched
    self.batchwise = batchwise

  def call(self, inputs):
    values = self.model(inputs, **self.kwargs)
    if self.goal:
      mask = goal_overlap(inputs, **self.kwargs)
      
      if self.minorder:
        minima = np.logical_and(
          mask,
          ndimage.minimum_filter(values, size=1+2*self.minorder, mode='constant') == values,
        )

        if np.any(minima):
          return np.argmin(np.where(minima, values, np.inf)), -np.where(mask, values, values[mask].max()+0.001)

      return np.argmin(np.where(mask, values, np.inf)), -np.where(mask, values, values[mask].max()+0.001)
    else:
      return np.argmin(values), -values



if False:

  def _apply_limit_and_exponent(inputs, **kwargs):
    limit = kwargs.get('limit', 0)
    exponent = kwargs.get('exponent', 1)

    if isinstance(limit, str):
      if limit=='mean':
        limit = inputs.mean()
      elif limit=='std':
        limit = inputs.mean() + inputs.std()
      else:
        raise ValueError('Invalid value {} for argument limit'.format(limit))



    if limit > 0:
      if limit < 1:
        inputs = np.maximum((inputs-limit)/(1-limit), 0)
      else:
        return np.where(inputs==1, 1., 0.)

    if exponent != 1:
      inputs **= exponent

    return inputs

  def _apply_scale(inputs, **kwargs):
    mask = kwargs.get('mask', None)
    if mask is not None:
      inmax = np.max(np.where(mask, inputs, -np.inf))
      inmin = np.min(np.where(mask, inputs, np.inf))
      inputs = np.where(mask, inputs, inmin)
    else:
      inmax = inputs.max()
      inmin = inputs.min()
    
    if inmax > inmin:
      return (inputs - inmin)/(inmax-inmin)
    else:
      return np.ones_like(inputs)

  def _goal_overlap(observation, previous=None, **kwargs):
    """Overlap between object and goal.

    Returned values are computed as
      max((overlap-limit)/(1-limit), 0)**exponent

    """
    g = np.where(observation[0][:,:,1]>0, 1, 0)
    t = np.where(observation[1][:,:,0]>0, 1, 0)
    overlap = signal.convolve2d(g, t, mode='valid')/np.count_nonzero(t)
    
    # Use different defaults for the goal overlap
    limit = kwargs.get('limit', 0.5)
    exponent = kwargs.get('exponent', 0.1)

    overlap = _apply_limit_and_exponent(overlap, limit=limit, exponent=exponent)

    if previous is not None:
      overlap *= previous
    return overlap
    
  def _random(observation, previous=None, **kwargs):
    return np.random.rand(
      observation[0].shape[0]-observation[1].shape[0]+1,
      observation[0].shape[1]-observation[1].shape[1]+1
    )*(previous if previous is not None else 1)

  def lowest(observation, previous=None, limit=0, exponent=1, **kwargs):
    x = observation[0][:,:,0]
    w = observation[1][:,:,0]

    shape = np.subtract(x.shape, w.shape) + 1
    h = np.zeros(shape)
    wbin = w > 0

    if previous is not None:
      # Use previous as a mask to compute only values that will be used
      if previous.shape != tuple(shape):
        previous.reshape(shape)
      mask = previous > 0
      irange = range(
        mask.argmax(),
        mask.size-np.argmax(np.flip(mask)),
      )
    else:
      irange = range(h.size)
      mask = None

    for idx in irange:
      i,j = np.unravel_index(idx, shape)  # pylint: disable=unbalanced-tuple-unpacking
      if mask is None or mask[i,j]:
        h[i,j] = np.max(np.where(
          wbin,
          x[i:i+w.shape[0], j:j+w.shape[1]] + w,
          0,
        ))

    h = _apply_scale(-h, mask=mask, **kwargs)
    h = _apply_limit_and_exponent(h, **kwargs)
    if previous is not None:
      h *= previous

    return h

  def wclosest(observation, previous=None, **kwargs):
    x = np.float32(observation[0][:,:,0])
    w = np.float32(observation[1][:,:,0])
    shape = np.subtract(x.shape, w.shape) + 1
    d = np.zeros(shape)

    wbin = w > 0
    wnz = np.count_nonzero(wbin)

    weighted = kwargs.get('weighted', True)
    if weighted:
      wi = np.arange(w.shape[0])
      wi *= wi[::-1]
      wj = np.arange(w.shape[1])
      wj *= wj[::-1]
      weights = np.where(
        wbin,
        np.expand_dims(wi, axis=1)*np.expand_dims(wj, axis=0),
        0,
      )
      weights = np.where(
        wbin,
        weights.max()-weights,
        0,
      )
      weights = weights/(weights.sum() or 1)
    else:
      weights = np.where(
        wbin,
        1/wnz,
        0.,
      )

    if previous is not None:
      if previous.shape != tuple(shape):
        previous.reshape(shape)
      mask = previous > 0
      irange = range(
        mask.argmax(),
        mask.size-np.argmax(np.flip(mask)),
      )
    else:
      irange = range(d.size)
      mask = None

    for idx in irange:
      i,j = np.unravel_index(idx, shape)  # pylint: disable=unbalanced-tuple-unpacking
      if mask is None or mask[i,j]:
        h = np.where(
          wbin,
          x[i:i+w.shape[0], j:j+w.shape[1]] + w,
          0.,
        )
        h = np.where(
          wbin,
          h.max() - h,
          0.,
        )
        h *= weights

        d[i,j] = np.sum(h)

    d = _apply_scale(-d, mask=mask, **kwargs)
    d = _apply_limit_and_exponent(d, **kwargs)

    if previous is not None:
      d *= previous

    return d

  def closest(observation, previous=None, **kwargs):
    x = np.float32(observation[0][:,:,0])
    w = np.float32(observation[1][:,:,0])
    shape = np.subtract(x.shape, w.shape) + 1
    d = np.zeros(shape)

    wbin = w > 0
    wnz = np.count_nonzero(wbin)

    if previous is not None:
      if previous.shape != tuple(shape):
        previous.reshape(shape)
      mask = previous > 0
      irange = range(
        mask.argmax(),
        mask.size-np.argmax(np.flip(mask)),
      )
    else:
      irange = range(d.size)
      mask = None

    for idx in irange:
      i,j = np.unravel_index(idx, shape)  # pylint: disable=unbalanced-tuple-unpacking
      if mask is None or mask[i,j]:
        h = np.where(
          wbin,
          x[i:i+w.shape[0], j:j+w.shape[1]] + w,
          0.,
        )
        h = np.where(
          wbin,
          h.max() - h,
          0.,
        )

        d[i,j] = np.sum(h)/wnz

    d = _apply_scale(-d, mask=mask, **kwargs)
    d = _apply_limit_and_exponent(d, **kwargs)

    if previous is not None:
      d *= previous

    return d

  try:
    import cv2 as cv

    def ccoeff(observation, previous=None, **kwargs):
      normed = kwargs.get('normed', True)

      img = observation[0][:,:,0]
      tmp = observation[1][:,:,0]
      c = cv.matchTemplate(  # pylint: disable=no-member
        img, 
        tmp, 
        cv.TM_CCOEFF_NORMED if normed else cv.TM_CCOEFF  # pylint: disable=no-member
      )
      c = _apply_scale(
        -c, 
        mask = (previous > 0 if previous is not None else None), 
        **kwargs
      )
      c = _apply_limit_and_exponent(c, **kwargs)

      if previous is not None:
        c *= previous
      
      return c

    def gradcorr_(observation, previous=None, **kwargs):
      normed = kwargs.get('normed', True)

      img = observation[0][:,:,0]
      tmp = observation[1][:,:,0]

      img_x = cv.Sobel(img, cv.CV_32F, 1, 0)  # pylint: disable=no-member
      img_y = cv.Sobel(img, cv.CV_32F, 0, 1)  # pylint: disable=no-member
      tmp_x = cv.Sobel(tmp, cv.CV_32F, 1, 0)  # pylint: disable=no-member
      tmp_y = cv.Sobel(tmp, cv.CV_32F, 0, 1)  # pylint: disable=no-member

      img = cv.merge([img_x, img_y])  # pylint: disable=no-member
      tmp = cv.merge([tmp_x, tmp_y])  # pylint: disable=no-member
      
      c = cv.matchTemplate(  # pylint: disable=no-member
        img, 
        tmp, 
        cv.TM_CCORR_NORMED if normed else cv.TM_CCORR  # pylint: disable=no-member
      )

      c = _apply_scale(
        -c, 
        mask = (previous > 0 if previous is not None else None), 
        **kwargs
      )
      c = _apply_limit_and_exponent(c, **kwargs)
      if previous is not None:
        c *= previous
      
      return c
  

    def goal_overlap_(observation, previous=None, **kwargs):
      """Overlap between object and goal.

      Returned values are computed as
        max((overlap-limit)/(1-limit), 0)**exponent

      """
      g = np.float32(observation[0][:,:,1])
      x = np.float32(observation[0][:,:,0])
      # g = np.where(g > 0, g, 0)
      t = np.float32(observation[1][:,:,0])
      c = cv.matchTemplate(  # pylint: disable=no-member
        g, 
        t, 
        cv.TM_CCORR,  # pylint: disable=no-member
      )
      c -= cv.matchTemplate(  # pylint: disable=no-member
        x, 
        t, 
        cv.TM_CCORR,  # pylint: disable=no-member
      )
      # c = _apply_scale(
      #   c, 
      #   mask = (previous > 0 if previous is not None else None), 
      #   **kwargs
      # )
      # Use different defaults for the goal overlap
      # limit = kwargs.get('limit', 0.5)
      # exponent = kwargs.get('exponent', 0.1)

      # c = _apply_limit_and_exponent(c, limit=limit, exponent=exponent)

      if previous is not None:
        c *= previous
      return c

  except ImportError:
    ccoeff = None
    gradcorr = None

  _methods = {
    'random': random,
    'lowest': lowest,
    'closest': closest,
    'wclosest':wclosest,
    'ccoeff': ccoeff,
    'corrcoef':corrcoef,
    'gradcorr': gradcorr
  }

  # @gin.configurable(module='stackrl')
  class _Baseline(agents.PyGreedy):
    def __init__(
      self, 
      method='random', 
      goal=True, 
      value=False, 
      unravel=False,
      batched=False,
      batchwise=False,
      **kwargs,
    ):
      if isinstance(method, str):
        if method=='none':
          method_list=[]
        else:
          method_list = method.lower().split('-')
          for i,m in enumerate(method_list):
            if m in methods:
              if methods[m] is not None:
                method_list[i] = methods[m]
              else:
                raise ImportError(
                  "opencv-python must be installed to use {} method.".format(m)
                )
            else:
              raise ValueError(
                "Invalid value {} for argument method. Must be in {}".format(method, methods)
              )
      elif callable(method):
        method_list = [method]
      else:
        raise TypeError(
          "Invalid type {} for argument method.".format(type(method))
        )

      def model(inputs):
        if goal:
          values = goal_overlap(inputs, **kwargs)
        else:
          values = None
        for m in method_list:
          values = m(inputs, previous=values, **kwargs)
        return values
      
      super(Baseline, self).__init__(
        model, 
        value=value, 
        unravel=unravel,
        batched=batched,
        batchwise=batchwise,
      )
