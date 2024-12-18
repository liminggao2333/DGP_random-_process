import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import gamma
import pandas as pd


# 核函数
def rbf_kernel(xs, ys, sigma=0.7, l=0.04):
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    tmp = (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)
    return tmp

# 学生 t 过程的 PDF
def student_t_pdf(y, nu, phi, kernel):
    n = len(y)
    diff = y - phi
    term1 = gamma((nu + n) / 2) / (gamma(nu / 2) * ((nu - 2) * np.pi) ** (n / 2))
    term2 = np.linalg.det(kernel) ** (-0.5)
    term3 = (1 + (diff.T @ np.linalg.pinv(kernel) @ diff) / (nu - 2)) ** (-(nu + n) / 2)
    return term1 * term2 * term3


# Metropolis-Hastings 抽样
def metropolis_hastings(num_samples, initial_value, proposal_sd, nu, phi, kernel):
    current_state = initial_value
    samples = np.zeros((num_samples, len(initial_value)))
    samples_quantity = 0
    for _ in range(num_samples):
        print(_)
        proposed_state = np.random.multivariate_normal(current_state, proposal_sd)
        cf_current = student_t_pdf(current_state, nu, phi, kernel)
        cf_proposed = student_t_pdf(proposed_state, nu, phi, kernel)
        acceptance_ratio = min(1, cf_proposed / cf_current)
        print(acceptance_ratio)
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
            samples[samples_quantity] = current_state
            samples_quantity += 1

    print(samples_quantity)
    return samples[0:samples_quantity, :]


# 双层 TP 实现
def deep_student_t_process(x,y, num_samples, nu1, nu2, kernel1_params, kernel2_params, proposal_sd):
    # 第一层：从第一个学生 t 过程采样 f
    k1 = rbf_kernel(x, x, **kernel1_params)
    phi1 = np.zeros(len(x))
    f_samples = metropolis_hastings(num_samples, np.ones(len(x))*y.mean(), proposal_sd, nu1, phi1, k1)

    # 第二层：给定 f，从第二个学生 t 过程采样 g
    f_mean = np.mean(f_samples, axis=0)
    k2 = rbf_kernel(f_mean, f_mean, **kernel2_params)
    phi2 = np.zeros(len(f_mean))
    g_samples = metropolis_hastings(num_samples, np.ones(len(f_mean))*y.mean(), proposal_sd, nu2, phi2, k2)

    g_samples = np.array(g_samples)
    return f_samples, g_samples

# 导入数据集
data = pd.read_excel('positive_negative_volatility_per_currency.xlsx', sheet_name = 'USD_JPY_Positive Volatility', header = None)
X = np.linspace(0, 10, 60)
y = data.iloc[3117: ,1].values  # 特征列
y=(y-y.min())/(y.max()-y.min())

train_size = int(len(X) * 0.5)  # 计算训练集的大小，即前50%的数据
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# 参数设置
num_samples = 1000
nu1, nu2 = 15, 15
kernel1_params = {'sigma': 0.5, 'l': 0.001}
kernel2_params = {'sigma': 0.5, 'l': 0.001}
proposal_sd = np.eye(len(X_test))*y_train.var()

# 执行双层 TP
f_samples, g_samples = deep_student_t_process(
    X_test,y_train, num_samples, nu1, nu2, kernel1_params, kernel2_params, proposal_sd
)


# 保存数据到 Excel
def save_to_excel(file_name, f_samples, g_samples):
    df_f = pd.DataFrame(f_samples, columns=[f"f_sample_{i}" for i in range(f_samples.shape[1])])
    df_g = pd.DataFrame(g_samples, columns=[f"g_sample_{i}" for i in range(g_samples.shape[1])])
    with pd.ExcelWriter(file_name) as writer:
        df_f.to_excel(writer, sheet_name='f_samples', index=False)
        df_g.to_excel(writer, sheet_name='g_samples', index=False)
    print(f"数据已保存到 {file_name}")


# 保存采样结果
save_to_excel("USD_JPY_Positive test result.xlsx", f_samples, g_samples)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(X_test, np.mean(f_samples, axis=0), label="Mean of f(t)", color='r')
plt.plot(X_test, np.mean(g_samples, axis=0), label="Mean of g(f)", color='b')
plt.fill_between(X_test, np.percentile(g_samples, 2.5, axis=0), np.percentile(g_samples, 97.5, axis=0), color='b', alpha=0.3,
                 label="95% Interval of g(f)")
plt.plot(X_test,y_test,label="True Value", color='g')
plt.legend()
plt.title("Deep Student-t Process: Two Layers")
plt.show()

y_pred = np.mean(g_samples, axis=0)
# 计算 MSE
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse}")
