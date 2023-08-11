import numpy as np
class MF():
    def __init__(self):
        # currently Do Nothing here
        # R, P, Q are the placeholder for matrix R, P, Q respectively
        # a 2-dimensional array(or list) is expected for each matrix
        self.R = None
        self.P = None
        self.Q = None
        
    def train(self, R, k, reg_lambda, lr_alpha, max_iterations, epsilon):
        # Write code here - begin
        # 이미지를 행렬 R로 사용
        self.R = R
        N = R.shape[0]
        M = R.shape[1]
        M_size_N=(N,k)
        M_size_M=(M,k)
        # 가우시안 분포에서 가중치를 초기화한 P와 Q 행렬 생성
        self.P = np.random.normal(scale=0.01, size=M_size_N)
        self.Q = np.random.normal(scale=0.01, size=M_size_M)

        # 주어진 파라미터로 행렬 분해 수행
        for step in range(max_iterations):
            for i in range(N):
                for j in range(M):
                    # 행렬 R의 0이 아닌 요소 만 고려
                    if R[i][j] > 0:
                        p_rating = R[i][j] - np.dot(self.P[i, :], self.Q[j, :].T)
                        self.P[i, :] = self.P[i, :] + lr_alpha * (2 * p_rating * self.Q[j, :] - reg_lambda * self.P[i, :])
                        self.Q[j, :] = self.Q[j, :] + lr_alpha * (2 * p_rating * self.P[i, :] - reg_lambda * self.Q[j, :])
            e = 0
            for i in range(N):
                for j in range(M):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(self.P[i, :], self.Q[j, :].T), 2)
                        e = e + (reg_lambda / 2) * (pow(np.linalg.norm(self.P[i, :]), 2) + pow(np.linalg.norm(self.Q[j, :]), 2))

            if(e<epsilon):
                break
        return e
    
    def P_MultipliedBy_Q_Transpose(self):
    # Write code here - begin
    # You should calculate "reconst_R = P * Q^T"
        reconst_R = np.dot(self.P, self.Q.T)

    # Write code here - end
    # a 2-dimensional array(or list) is expected for reconst_R
        return reconst_R