import torch
import torch.nn.functional as F
from .common import approx_hessian


def dc_expansion_g_adp(w_t_g, w_t_tau, grad, lmd, lr):
    """
    补偿模型梯度并自适应更新 lambda 系数。
    
    计算公式:
    g_{w_{t+\tau}} = g_{w_t} + lmd * (w_t_tau - w_t_g) * H_{w_t_g}
    
    同时计算自适应 lambda 更新值:
    next_lmd_const = ||g'||_2 / ||H * (w_t_tau - w_t_g)||_2
    
    参数:
        w_t_g: 全局模型参数 (原始时刻 t)
        w_t_tau: 延迟后的模型参数 (t+τ 时刻)
        grad: 在全局模型上计算的梯度
        lmd: 当前的 lambda 值
        lr: 学习率
        
    返回:
        next_lmd_const: 下一步的 lambda 系数，下一步 lmd *= next_lmd_const
    """
    # 计算Hessian矩阵的近似值
    hessian_approx = approx_hessian(grad)
    next_lmd_up, next_lmd_down = 0.0, 0.0
    
    # 更新模型
    for p_t, p_t_tau, g, h_a in zip(w_t_g, w_t_tau, grad, hessian_approx):
        # 计算补偿后的梯度
        tmp_g = g + lmd * (p_t_tau.data - p_t.data) * h_a
        
        # 累计自适应 lambda 需要的二范数
        next_lmd_up += torch.sum(torch.pow(g, 2))
        next_lmd_down += torch.sum(torch.pow(h_a * (p_t_tau.data - p_t.data), 2))
        
        # 更新参数
        p_t_tau.data -= lr * tmp_g
    
    # 计算新的 lambda 值，添加数值稳定性保护
    if next_lmd_down > 1e-10:
        next_lmd_const = torch.sqrt(next_lmd_up) / torch.sqrt(next_lmd_down)
        # 添加上下界限制，防止数值不稳定
        next_lmd_const = torch.clamp(next_lmd_const, 0.01, 100.0)
    else:
        next_lmd_const = 1
        
    return next_lmd_const

def dc_expansion_g(w_t_g, w_t_tau, grad, lmd, lr):
    """
    这个测试函数将模型的梯度进行补偿。
    计算公式为：
    g_{w_{t+\tau}} = g_{w_t} + (w_t_tau - w_t_g) \cdot H_{w_t_g} （忽略参数的三阶导数）
    g_t 是模型参数 w_t_g 在 t 时刻的梯度。
    dg/dt 是模型参数 w_t_g 在 t 时刻的二阶导数，即Hessian矩阵。
    同样使用Fisher信息矩阵的近似值来代替Hessian矩阵。
    """
    # 计算Hessian矩阵的近似值
    hessian_approx = approx_hessian(grad)
    # 更新模型
    for p_t, p_t_tau, g, h_a in zip(w_t_g, w_t_tau, grad, hessian_approx):
        # 计算补偿量
        tmp_g = g + lmd * (p_t_tau.data - p_t.data) * h_a
        # 更新参数
        p_t_tau.data -= lr * tmp_g

def dc_braindead(w_t, w_t_tau, w_t_g):
    """
    在一切都是iid的情况下，
    w_t - w_t^{global} = w_t_tau - w_t_tau^{global}
    则仿造出来的 w_t_tau^{global} = w_t_tau - (w_t - w_t^{global})。
    """
    with torch.no_grad():
        for p_t, p_t_tau, p_t_global in zip(w_t, w_t_tau, w_t_g):
            p_t_tau.data = p_t_tau.data - (p_t.data - p_t_global.data)

def dc_vanilla(w_t_tau, w_t_g):
    """
    直接把 w_t_tau 设置为 w_t_g。
    """
    with torch.no_grad():
        for p_t_tau, p_t_global in zip(w_t_tau, w_t_g):
            p_t_tau.data = p_t_global.data
    
def dc_streaming_diloco(w_t_g, w_t_tau, alpha):
    """
    将收到的全局模型 w_t_g 和 w_t_tau 进行线性加权平均。
    计算公式为：
    w_t_tau = alpha * w_t + (1 - alpha) * w_t_tau
    """
    with torch.no_grad():
        for p_t, p_t_tau in zip(w_t_g, w_t_tau):
            p_t_tau.data = alpha * p_t.data + (1 - alpha) * p_t_tau.data
    
def dc_with_taylor_expansion_at_w(w_t_g, w_t_tau, tau, grad, lmd):
    """
    没找到数学上的意义
    这个测试函数将使用模型参数 W_{t+\tau} 在 t+\tau 时刻的泰勒展开来计算延迟补偿。
    计算公式为：
    W_{t+\tau+1} = w_t_tau - \eta * 
    其中 dW/dt 和 d^2W/dt^2 分别是模型参数 w_t_g 在 t 时刻的一阶和二阶导数，
    即梯度和Hessian矩阵。
    由于Hessian矩阵计算复杂度很高，可以采用Fisher信息矩阵的计算方式代替。
    这里我们使用Fisher信息矩阵的近似值来代替Hessian矩阵。
    计算方式：
    Fisher = E[(dW/dt)^2]
    其中 E 是期望值。
    返回值为延迟补偿后的模型参数 W_{t+\tau}。
    """
    hessian_approx = approx_hessian(lmd, grad)
    for p_t, p_t_tau, h_a in zip(w_t_g, w_t_tau, hessian_approx):
        # 估计一阶导数（参数变化率）
        first_order = (p_t_tau.data - p_t.data) / tau
        # 泰勒展开补偿
        compensated = p_t.data + tau * first_order + 0.5 * tau**2 * h_a
        p_t_tau.data = compensated