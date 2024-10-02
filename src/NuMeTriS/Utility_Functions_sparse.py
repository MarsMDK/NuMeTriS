import numpy as np
from numba import jit, prange, float64
from scipy.optimize import least_squares as lsq





def ll_DBCM(params,k_out,k_in):

    """Opposite Log-likelihood function for the DBCM, saved in self.ll"""
    
    
    n = len(k_out)
    a_out = params[:n]
    a_in = params[n:]
    x_out = np.exp(-a_out)
    x_in = np.exp(-a_in)

    ll = 0
    for i in range(n):
        ll += - k_out[i]*a_out[i] - k_in[i]*a_in[i] 
    
        for j in range(n):
            Gij = x_out[i]*x_in[j]    
            ll -= np.log(1.+Gij)

    
    return  - ll



def jac_DBCM(params,k_out,k_in):

    """Opposite Jacobian for the DBCM, saved in self.jacobian"""
    
    n = len(k_out)
    
    a_out = params[:n]
    a_in = params[n:]
    x_out = np.exp(-a_out)
    x_in = np.exp(-a_in)
    jac = np.empty(len(params))
    
    for i in range(n):
        jac[i] = - k_out[i]
        jac[i+n] = - k_in[i]
        for j in range(n):
            if j !=i:
            
                Gij = x_out[i]*x_in[j]    
                pij = Gij/(1.+Gij)
                Gji = x_out[j]*x_in[i]    
                pji = Gji/(1.+Gji)
            
            
                jac[i] += pij
                jac[i+n] += pji

    return  - jac


def relative_error_DBCM(params,k_out,k_in):

    """Relative error function for the DBCM, saved in self.relative_error."""
    
    n = len(k_out)
    
    a_out = params[:n]
    a_in = params[n:]
    x_out = np.exp(-a_out)
    x_in = np.exp(-a_in)
    jac = np.empty(len(params))
    normalization = np.empty(len(params))
    
    for i in range(n):
        jac[i] = - k_out[i]
        jac[i+n] = - k_in[i]
        normalization[i] = - k_out[i]
        normalization[i+n] = - k_in[i]
        
        for j in range(n):
            if j !=i:
            
                Gij = x_out[i]*x_in[j]    
                pij = Gij/(1.+Gij)
                Gji = x_out[j]*x_in[i]    
                pji = Gji/(1.+Gji)
            
            
                jac[i] += pij
                jac[i+n] += pji
    for i in range(n):
        if normalization[i]!=0:
            jac[i]/=normalization[i]
    
    return  - jac



def hess_DBCM(params,k_out,k_in):

    """Computes opposite of the Hessian function for the DBCM.
    
    :param params: parameter value for the computation of hessian.
    :type params: np.ndarray

    :param k_out: empirical out-degree.
    :type k_out: np.ndarray

    :param k_in: empirical in-degree.
    :type k_in: np.ndarray

    :return: Opposite of Hessian.
    :rtype: np.ndarray

    """
    
    n = len(k_out)
    
    a_out = params[:n]
    a_in = params[n:]
    x_out = np.exp(-a_out)
    x_in = np.exp(-a_in)
    hess = np.zeros((len(params),len(params)))
    
    for i in range(n):
        for j in range(n):

            if j==i:
                
                for k in range(n):
                    if k!=i:

                        Gik = x_out[i]*x_in[k]    
                        pik = Gik/(1.+Gik)

                        Gki = x_out[k]*x_in[i]    
                        pki = Gki/(1.+Gki)

                        hess[i,i] += -pik*(1.-pik)
                        hess[i+n,i+n] += - pki*(1.-pki)
                    
            else:
                
                Gij = x_out[i]*x_in[j]    
                pij = Gij/(1.+Gij)
                Gji = x_out[j]*x_in[i]    
                pji = Gji/(1.+Gji)
                

                hess[i,j+n] = - pij*(1.-pij)
                hess[j+n,i] = - pij*(1.-pij)
                hess[j,i+n] = - pji*(1.-pji)        
                hess[i+n,j] = - pji*(1.-pji)        
                
    return  - hess




def ll_RBCM(params,k_right,k_left,k_rec):
    
    """Opposite Log-likelihood function for the RBCM, saved in self.ll"""
    
    N = len(k_right)
    a_right = params[:N]
    a_left = params[N:2*N]
    a_rec = params[2*N:3*N]
    
    x = np.exp(-a_right)
    y = np.exp(-a_left)
    z = np.exp(-a_rec)
    ll = 0
    for i in range(N):
        ll += -a_right[i]*k_right[i] - a_left[i]*k_left[i] - a_rec[i]*k_rec[i]
        for j in range(N):
            if j>i:
            
                aux_ij = x[i]*y[j]
                aux_ji = x[j]*y[i]
                aux_und = z[i]*z[j]
                partition = 1.+ aux_ij + aux_ji + aux_und
                
                ll += - np.log(partition)
                
    return  - ll


def jac_RBCM(params,k_right,k_left,k_rec):
    """Opposite jacobian function for the RBCM, saved in self.ll"""
    
    
    N = int(len(params)/3)
    a_right = params[:N]     
    a_left = params[N:2*N]
    a_rec = params[2*N:3*N]
    
    x = np.exp(-a_right)
    y = np.exp(-a_left)
    z = np.exp(-a_rec)
    
    jac = np.empty(len(params))
    
    for i in range(N):
        jac[i] = - k_right[i]
        jac[i+N] = - k_left[i]
        jac[i+2*N] = - k_rec[i]
        for j in range(N):
            if j!=i:
                
                aux_ij = x[i]*y[j]
                aux_ji = x[j]*y[i]
                aux_und = z[i]*z[j]
                aux = aux_ij + aux_ji + aux_und

                pij_right = aux_ij/(1.+aux)
                pij_left = aux_ji/(1.+aux)
                pij_rec = aux_und/(1.+aux)

                jac[i] += pij_right
                jac[i+N] += pij_left
                jac[i+2*N] += pij_rec
    
    return - jac


def relative_error_RBCM(params,k_right,k_left,k_rec):
    """Relative error in the solution of the RBCM, saved in self.relative_error"""
    
    
    N = int(len(params)/3)
    a_right = params[:N]
    a_left = params[N:2*N]
    a_rec = params[2*N:3*N]
    
    x = np.exp(-a_right)
    y = np.exp(-a_left)
    z = np.exp(-a_rec)
    
    jac = np.empty(len(params))
    normalization = np.empty(len(params))
    for i in range(N):
        jac[i] = - k_right[i]
        jac[i+N] = - k_left[i]
        jac[i+2*N] = - k_rec[i]
        normalization[i] = - k_right[i]
        normalization[i+N] = - k_left[i]
        normalization[i+2*N] = - k_rec[i]
        
        for j in range(N):
            if j!=i:
                ij = i*N+j 
                ji = j*N+i

                aux_ij = x[i]*y[j]
                aux_ji = x[j]*y[i]
                aux_und = z[i]*z[j]
                aux = aux_ij + aux_ji + aux_und

                pij_right = aux_ij/(1.+aux)
                pij_left = aux_ji/(1.+aux)
                pij_rec = aux_und/(1.+aux)

                jac[i] += pij_right
                jac[i+N] += pij_left
                jac[i+2*N] += pij_rec
    
    for i in range(N):
        if normalization[i] != 0:
            jac[i]/=normalization[i]

    return - jac



def hess_RBCM(params,k_right,k_left,k_rec):
    """Computes opposite of the Hessian function for the RBCM.
    
    :param params: Parameters for the solution of RBCM.
    :type params: np.ndarray

    :param k_right: Empirical non-reciprocated out-degree.
    :type k_right: np.ndarray

    :param k_left: Empirical non-reciprocated in-degree.
    :type k_left: np.ndarray

    :param k_rec: Empirical reciprocated degree.
    :type k_rec: np.ndarray

    :return: Opposite of the Hessian function for the RBCM
    :rtype: np.ndarray
    """
    
    
    n = int(len(params)/3)

    x = np.exp(-params[:n])
    y = np.exp(-params[n:2*n])
    z = np.exp(-params[2*n:3*n])
    
    hess = np.zeros((len(params),len(params)))
    
    for i in range(n):
        
        for j in range(n):
            if j==i:

                for k in range(n):
                    if k!=i:
                        aux_ik = x[i]*y[k]
                        aux_ki = x[k]*y[i]
                        aux_und = z[i]*z[k]
                        aux = aux_ik + aux_ki + aux_und

                        pik_right = aux_ik/(1.+aux)
                        pik_left = aux_ki/(1.+aux)
                        pik_rec = aux_und/(1.+aux)

                        
                        hess[i,i] += - pik_right*(1.-pik_right)
                        hess[i,i+n] += pik_right*pik_left
                        hess[i,i+2*n] += pik_right*pik_rec
                        
                        hess[i+n,i] += pik_right*pik_left
                        hess[i+n,i+n] += - pik_left*(1.-pik_left)
                        hess[i+n,i+2*n] += pik_left*pik_rec
                        
                        hess[i+2*n,i] +=pik_rec*pik_right
                        hess[i+2*n,i+n] +=pik_rec*pik_left
                        hess[i+2*n,i+2*n] += - pik_rec*(1.-pik_rec)
                        
            else:

                aux_ij = x[i]*y[j]
                aux_ji = x[j]*y[i]
                aux_und = z[i]*z[j]
                aux = aux_ij + aux_ji + aux_und

                pij_right = aux_ij/(1.+aux)
                pij_left = aux_ji/(1.+aux)
                pij_rec = aux_und/(1.+aux)

                
                hess[i,j] = pij_right*pij_left
                hess[i,n+j] = -pij_right*(1.-pij_right)
                hess[i,2*n+j] = pij_right*pij_rec

                hess[n+i,j] = -pij_left*(1.-pij_left)
                hess[n+i,n+j] = pij_right*pij_left
                hess[n+i,2*n+j] = pij_left*pij_rec

                hess[2*n+i,j] = pij_left*pij_rec
                hess[2*n+i,n+j] = pij_right*pij_rec
                hess[2*n+i,2*n+j] = -pij_rec*(1.-pij_rec)               

    return - hess


# @jit(nopython=True,fastmath=True)
def ll_CReMa_after_DBCM(params,s_out,s_in,params_DBCM):

    """Computes opposite of the log-likelihood function for the conditional weighted model CReMa after the solution of DBCM."""

    n = len(s_out)
    b_out = params[:n]
    b_in = params[n:2*n]
    x_out = np.exp(-params_DBCM[:n])
    x_in = np.exp(-params_DBCM[n:2*n])
    
    ll = 0.
    for i in range(n):
        ll += - b_out[i]*s_out[i] - b_in[i]*s_in[i]
        
        for j in range(n):
            if j!=i:
                
                pij = x_out[i]*x_in[j]/(1.+x_out[i]*x_in[j])
                
                ll += pij*np.log(b_out[i]+b_in[j])
            
    return - ll

# @jit(nopython=True,fastmath=True)
def jac_CReMa_after_DBCM(params,s_out,s_in,params_DBCM):

    """Computes opposite of the jacobian function for the conditional weighted model CReMa after the solution of DBCM."""

    n = int(len(params)/2)
    b_out = params[:n]
    b_in = params[n:2*n]
    x_out = np.exp(-params_DBCM[:n])
    x_in = np.exp(-params_DBCM[n:2*n])
    
    #fare ll
    jac = np.empty(len(params))
    for i in range(n):
        jac[i] = - s_out[i]
        jac[i+n] = - s_in[i]
        
        
        for j in range(n):
            if j!=i:
                
                pij = x_out[i]*x_in[j]/(1.+x_out[i]*x_in[j])
                pji = x_out[j]*x_in[i]/(1.+x_out[j]*x_in[i])

                jac[i] += pij/(b_in[j]+b_out[i])
            
                jac[i+n] += pji/(b_in[i]+b_out[j])
            
                
    return - jac

# @jit(nopython=True,fastmath=True)
def relative_error_CReMa_after_DBCM(params,s_out,s_in,params_DBCM):

    """Computes opposite of the relative_error function for the conditional weighted model CReMa after the solution of DBCM."""

    n = int(len(params)/2)
    b_out = params[:n]
    b_in = params[n:2*n]
    x_out = np.exp(-params_DBCM[:n])
    x_in = np.exp(-params_DBCM[n:2*n])
    
    
    jac = np.empty(len(params))
    normalization = np.empty(len(params))

    for i in range(n):
        jac[i] = - s_out[i]
        jac[i+n] = - s_in[i]
        normalization[i] = - s_out[i]
        normalization[i+n] = - s_in[i]
        
        for j in range(n):
            if j!=i:
                
                pij = x_out[i]*x_in[j]/(1.+x_out[i]*x_in[j])
                pji = x_out[j]*x_in[i]/(1.+x_out[j]*x_in[i])

                jac[i] += pij/(b_in[j]+b_out[i])
            
                jac[i+n] += pji/(b_in[i]+b_out[j])
            
    for i in range(n):
        if normalization[i] != 0:
            jac[i]/=normalization[i]

    return - jac

# @jit(nopython=True)
def hess_CReMa_after_DBCM(params,s_out,s_in,params_DBCM):

    """Computes opposite of the hessian function for the conditional weighted model CReMa after the solution of DBCM.
    
    :param params: Parameters for the CReMa after the solution of DBCM.
    :type params: np.ndarray

    :param s_out: Empirical out-strength.
    :type s_out: np.ndarray
    
    :param s_in: Empirical in-strength.
    :type s_in: np.ndarray

    :param params_DBCM: Parameters for the solution of DBCM.
    :type params_DBCM: np.ndarray
    
    :return: Opposite of the hessian function for the CReMa model, after the solution of DBCM
    :rtype: np.ndarray
    """

    n = int(len(params)/2)
    b_out = params[:n]
    b_in = params[n:]
    x_out = np.exp(-params_DBCM[:n])
    x_in = np.exp(-params_DBCM[n:])
    
    hess = np.zeros((2*n,2*n))
    for i in range(n):
            
        for j in range(n):
            if j==i:
                for k in range(n):
                    if k != i:

                        
                        pik = x_out[i]*x_in[k]/(1.+x_out[i]*x_in[k])
                        pki = x_out[k]*x_in[i]/(1.+x_out[k]*x_in[i])


                        
                        hess[i,j] += - pik/(b_out[i]+b_in[k])**2
                        hess[i+n,i+n] += - pki/(b_out[k]+b_in[i])**2
                
            else:
                
                aij = x_out[i]*x_in[j]/(1.+x_out[i]*x_in[j])
                aji = x_out[j]*x_in[i]/(1.+x_out[j]*x_in[i])

                hess[i,j+n] = - aij/(b_in[j]+b_out[i])**2
                hess[i+n,j] = - aji/(b_in[i]+b_out[j])**2
                
    return - hess


# @jit(nopython=True,fastmath=True)
def ll_CRWCM_after_RBCM(params,s_right,s_left,s_rec_out,s_rec_in,params_RBCM):

    """Computes opposite of the log-likelihood function for the conditional weighted model CRWCM after the solution of RBCM.
    Saved in self.ll."""

    n = int(len(params)/4)
    b_right=params[:n]
    b_left=params[n:2*n]
    b_rec_out=params[2*n:3*n]
    b_rec_in=params[3*n:]

    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])

    ll = 0.
    for i in range(n):
        ll += -b_right[i]*s_right[i] - b_left[i]*s_left[i]-b_rec_out[i]*s_rec_out[i] - b_rec_in[i]*s_rec_in[i]

        for j in range(n):
            if j>i:
                
                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                
                aij_right = aux_p_right/aux_p
                aij_left = aux_p_left/aux_p
                aij_rec = aux_p_rec/aux_p

                aux_right = (b_right[i]+b_left[j])
                aux_left = (b_left[i]+b_right[j])
                aux_rec_out =(b_rec_out[i]+b_rec_in[j])
                aux_rec_in = (b_rec_in[i]+b_rec_out[j])

                lnZ = aij_right*np.log(aux_right) + aij_left*np.log(aux_left) + aij_rec*np.log(aux_rec_out) + aij_rec*np.log(aux_rec_in)

                ll += lnZ
        
    return - ll

# @jit(nopython=True,fastmath=False)
def ll_CRWCM_after_RBCM_rl(params,s_right,s_left,params_RBCM):

    """Computes opposite of the log-likelihood function for the conditional weighted model CRWCM after the solution of RBCM for the
    non-reciprocated sub-problem."""

    n = int(len(params)/2)
    b_right=params[:n]
    b_left=params[n:2*n]
    
    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])

    ll = 0.
    for i in range(n):
        ll += -b_right[i]*s_right[i] - b_left[i]*s_left[i]

        for j in range(n):
            if j!=i:
                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                
                aij_right = aux_p_right/aux_p
                aij_left = aux_p_left/aux_p
                aij_rec = aux_p_rec/aux_p

                aux_right = (b_right[i]+b_left[j])
                aux_left = (b_left[i]+b_right[j])
                
                lnZ = aij_right*np.log(aux_right) #+ aij_left*np.log(aux_left) 
                
                ll += lnZ
    
    return - ll

# @jit(nopython=True,fastmath=False)
def ll_CRWCM_after_RBCM_rec(params,s_rec_out,s_rec_in,params_RBCM):

    """Computes opposite of the log-likelihood function for the conditional weighted model CRWCM after the solution of RBCM for the
    reciprocated sub-problem."""

    n = int(len(params)/2)
    b_rec_out=params[:n]
    b_rec_in=params[n:]

    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])

    ll = 0.
    for i in range(n):
        ll += -b_rec_out[i]*s_rec_out[i] - b_rec_in[i]*s_rec_in[i]

        for j in range(n):
            if j!=i:

                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec
    
                aij_rec = aux_p_rec/aux_p

                aux_rec_out =(b_rec_out[i]+b_rec_in[j])
                aux_rec_in = (b_rec_in[i]+b_rec_out[j])

                lnZ = aij_rec*np.log(aux_rec_out) #+ aij_rec*np.log(aux_rec_in)
                
                ll += lnZ
        
    return - ll


# @jit(nopython=True,fastmath=True)
def jac_CRWCM_after_RBCM(params,s_right,s_left,s_rec_out,s_rec_in,params_RBCM):

    """Computes opposite of the jacobian function for the conditional weighted model CRWCM after the solution of RBCM.
    Saved in self.jacobian."""

    n = int(len(params)/4)
    b_right=params[:n]
    b_left=params[n:2*n]
    b_rec_out=params[2*n:3*n]
    b_rec_in=params[3*n:]

    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])


    jac = np.zeros(len(params))
    for i in range(n):
        jac[i] = - s_right[i]
        jac[i+n] = - s_left[i]
        jac[i+2*n] = - s_rec_out[i]
        jac[i+3*n] = - s_rec_in[i]
        
        
        for j in range(n):
            if j!=i:
                ij = i*n+j
                ji = j*n+i

                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                
                aij_right = aux_p_right/aux_p
                aij_left = aux_p_left/aux_p
                aij_rec = aux_p_rec/aux_p

                jac[i] += aij_right/(b_right[i]+b_left[j])
                jac[i+n] += aij_left/(b_left[i]+b_right[j])
                jac[i+2*n] += aij_rec/(b_rec_out[i]+b_rec_in[j])
                jac[i+3*n] += aij_rec/(b_rec_in[i]+b_rec_out[j])

                
    return - jac

# @jit(nopython=True,fastmath=True)
def relative_error_CRWCM_after_RBCM(params,s_right,s_left,s_rec_out,s_rec_in,params_RBCM):

    """Computes opposite of the relative error function for the conditional weighted model CRWCM after the solution of RBCM. 
    Saved in self.relative_error."""

    n = int(len(params)/4)
    b_right=params[:n]
    b_left=params[n:2*n]
    b_rec_out=params[2*n:3*n]
    b_rec_in=params[3*n:]

    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])


    jac = np.empty(len(params))
    normalization = np.empty(len(params))
    
    for i in range(n):
        jac[i] = - s_right[i]
        jac[i+n] = - s_left[i]
        jac[i+2*n] = - s_rec_out[i]
        jac[i+3*n] = - s_rec_in[i]
        normalization[i] = - s_right[i]
        normalization[i+n] = - s_left[i]
        normalization[i+2*n] = - s_rec_out[i]
        normalization[i+3*n] = - s_rec_in[i]
        
        
        for j in range(n):
            if j!=i:
                ij = i*n+j
                ji = j*n+i

                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                
                aij_right = aux_p_right/aux_p
                aij_left = aux_p_left/aux_p
                aij_rec = aux_p_rec/aux_p

                jac[i] += aij_right/(b_right[i]+b_left[j])
                jac[i+n] += aij_left/(b_left[i]+b_right[j])
                jac[i+2*n] += aij_rec/(b_rec_out[i]+b_rec_in[j])
                jac[i+3*n] += aij_rec/(b_rec_in[i]+b_rec_out[j])

    for i in range(4*n):
        if normalization[i] != 0:
            jac[i]/=normalization[i]

                
    return - jac


# @jit(nopython=True,fastmath=True)
def jac_CRWCM_after_RBCM_rl(params,s_right,s_left,params_RBCM):

    """Computes opposite of the jacobian function for the conditional weighted model CRWCM after the solution of RBCM,
    for the non-reciprocated sub-problem."""

    n = int(len(params)/2)
    b_right=params[:n]
    b_left=params[n:2*n]

    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])

    
    jac = np.empty(len(params))
    for i in range(n):
        jac[i] = - s_right[i]
        jac[i+n] = - s_left[i]
        
        
        for j in range(n):
            if j!=i:
                ij = i*n+j
                ji = j*n+i

                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                
                aij_right = aux_p_right/aux_p
                aij_left = aux_p_left/aux_p
                aij_rec = aux_p_rec/aux_p

                jac[i] += aij_right/(b_right[i]+b_left[j])
                jac[i+n] += aij_left/(b_left[i]+b_right[j])
            
                
    return - jac


# @jit(nopython=True,fastmath=True)
def hess_CRWCM_after_RBCM_rl(params,s_right,s_left,params_RBCM):

    """Computes opposite of the hessian function for the conditional weighted model CRWCM after the solution of RBCM,
    for the non-reciprocated sub-problem."""

    n = int(len(params)/2)
    b_right=params[:n]
    b_left=params[n:2*n]

    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])

    
    hess = np.zeros((len(params),len(params)))
    for i in range(n):
        
        
        for j in range(n):
            if j!=i:
                ij = i*n+j
                ji = j*n+i

                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                
                aij_right = aux_p_right/aux_p
                aij_left = aux_p_left/aux_p
                aij_rec = aux_p_rec/aux_p

                bij_right2 = (b_right[i] + b_left[j])**2
                bij_left2 = (b_right[j]+b_left[i])**2


                hess[i,n+j] =  - aij_right/bij_right2
                hess[i+n,j] =  - aij_left/bij_left2

            else:
                for k in range(n):
                    if k!=i:
           
                        aux_p_right = x_right[i]*x_left[k]
                        aux_p_left = x_right[k]*x_left[i]
                        aux_p_rec = x_rec[i]*x_rec[k]
                        aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec
             
                        aik_right = aux_p_right/aux_p
                        aik_left = aux_p_left/aux_p
                        aik_rec = aux_p_rec/aux_p

                        bik_right2 = (b_right[i] + b_left[k])**2
                        bik_left2 = (b_right[k]+b_left[i])**2
        
                        
                        hess[i,i] += - aik_right/bik_right2     
                        hess[i+n,i+n] += - aik_left/bik_left2
                                            
    return - hess


# @jit(nopython=True,fastmath=True)
def jac_CRWCM_after_RBCM_rec(params,s_rec_out,s_rec_in,params_RBCM):

    """Computes opposite of the jacobian function for the conditional weighted model CRWCM after the solution of RBCM,
    for the reciprocated sub-problem."""

    n = int(len(params)/2)
    b_rec_out=params[:n]
    b_rec_in=params[n:]

    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])


    jac = np.empty(len(params))
    for i in range(n):
        jac[i] = - s_rec_out[i]
        jac[i+n] = - s_rec_in[i]
        
        
        for j in range(n):
            if j!=i:
                ij = i*n+j
                ji = j*n+i

                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                
                aij_right = aux_p_right/aux_p
                aij_left = aux_p_left/aux_p
                aij_rec = aux_p_rec/aux_p

                
                jac[i] += aij_rec/(b_rec_out[i]+b_rec_in[j])
                jac[i+n] += aij_rec/(b_rec_in[i]+b_rec_out[j])
        
    

                
    return - jac

# @jit(nopython=True,fastmath=True)
def hess_CRWCM_after_RBCM_rec(params,s_rec_out,s_rec_in,params_RBCM):

    """Computes opposite of the hessian function for the conditional weighted model CRWCM after the solution of RBCM,
    for the reciprocated sub-problem."""
    
    
    n = int(len(params)/2)
    b_rec_out=params[:n]
    b_rec_in=params[n:2*n]

    x_right = np.exp(-params_RBCM[:n])
    x_left = np.exp(-params_RBCM[n:2*n])
    x_rec = np.exp(-params_RBCM[2*n:3*n])

    
    hess = np.zeros((len(params),len(params)))
    for i in range(n):
        
        
        for j in range(n):
            if j!=i:
                ij = i*n+j
                ji = j*n+i

                aux_p_right = x_right[i]*x_left[j]
                aux_p_left = x_right[j]*x_left[i]
                aux_p_rec = x_rec[i]*x_rec[j]
                aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                
                aij_right = aux_p_right/aux_p
                aij_left = aux_p_left/aux_p
                aij_rec = aux_p_rec/aux_p

                
                
                delta_ij_1 = aij_rec/(b_rec_out[i]+b_rec_in[j])**2
                delta_ji_1 = aij_rec/(b_rec_in[i]+b_rec_out[j])**2

                hess[i,n+j] = - delta_ij_1
                hess[i+n,j] = - delta_ji_1

            else:
                for k in range(n):
                    if k!=i:
                        ik = i*n+k
                        ki = k*n+i

                        aux_p_right = x_right[i]*x_left[k]
                        aux_p_left = x_right[k]*x_left[i]
                        aux_p_rec = x_rec[i]*x_rec[k]
                        aux_p = 1.+ aux_p_right + aux_p_left + aux_p_rec

                        
                        aik_right = aux_p_right/aux_p
                        aik_left = aux_p_left/aux_p
                        aik_rec = aux_p_rec/aux_p
                        
                        
                        delta_ik_1 = aik_rec/(b_rec_out[i]+b_rec_in[k])**2
                        delta_ki_1 = aik_rec/(b_rec_in[i]+b_rec_out[k])**2

                        hess[i,i] += - delta_ik_1
                        hess[i+n,i+n] += -delta_ki_1
                                            
    return -hess




# @jit(nopython=True,fastmath=True)
def armijo_condition(
    f_old, f_new, alpha, grad_f, p, c1=1e-04
):
    """return boolean indicator if armijo wolfe condition are respected."""
    grad_f = np.ascontiguousarray(grad_f)
    pT = np.ascontiguousarray(p.T)
    sup = f_old + c1 * alpha * grad_f @ pT
    
    
    return bool(f_new < sup)

# @jit(nopython=True,fastmath=True)
def curvature_condition(
    grad_f, grad_f_new, p, c2=0.9
):
    """return boolean indicator if curvature wolfe condition are respected."""
    grad_f = np.ascontiguousarray(grad_f)
    grad_f_new = np.ascontiguousarray(grad_f_new)
    
    pT = np.ascontiguousarray(p.T)
    
    a = grad_f_new @ pT
    b = c2 * grad_f @ pT
    
    return bool(a >= b)

# @jit(nopython=True,fastmath=True)
def linsearch_fun_DBCM(X,args):

    """Function for the linear search of optimal parameter for Newton optimisation of DBCM."""
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    jac = X[4]
    step_fun = X[5]
        
        
    i = 0
    s_old = step_fun(x,*args)
    f = jac(x,*args)
    while (
        (armijo_condition(
            s_old, step_fun(x + alfa * dx,*args), alfa, f, dx
        )
        == False and i < 50)):#or (curvature_condition(f,jac(x+alfa*dx,*args), dx)==False) :
    
        alfa *= beta        
        i += 1

    return alfa


# @jit(nopython=True,fastmath=True)
def linsearch_fun_RBCM(X,args):

    """Function for the linear search of optimal parameter for Newton optimisation of RBCM."""
    
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    jac = X[4]
    step_fun = X[5]
        
        
    i = 0
    s_old = step_fun(x,*args)
    f = jac(x,*args)
    while (
        (armijo_condition(
            s_old, step_fun(x + alfa * dx,*args), alfa, f, dx
        )
        == False and i < 20)):#or (curvature_condition(f,jac(x+alfa*dx,*args), dx)==False) :
    
        alfa *= beta        
        i += 1

    return alfa

# @jit(nopython=True,fastmath=True)
def linsearch_fun_CRWCM_separated(X,args):

    """Function for the linear search of optimal parameter for Newton optimisation of CRWCM and CReMa."""
    
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    jac = X[4]
    step_fun = X[5]
    linsteps = 0
    n = int(len(x)/2.)
    while True:
        linsteps += 1
    
        next_lam = (x+alfa*dx)[:n]
        next_eps = (x+alfa*dx)[n:2*n]
        
        lam_min_index = next_lam.argsort()[:2]
        eps_min_index = next_eps.argsort()[:2]
        
        cond_lami_epsj = False
        
        if lam_min_index[0] == eps_min_index[0]:
            cond1 = next_lam[lam_min_index[0]] + next_eps[eps_min_index[1]] > 0.
            cond2 = next_lam[lam_min_index[1]] + next_eps[eps_min_index[0]] > 0.
            cond_lami_epsj = cond1 and cond2     
        else:
            cond_lami_epsj = next_lam[lam_min_index[0]] + next_eps[eps_min_index[0]] > 0.

        
        if  cond_lami_epsj:
            break
        else:
            alfa *= beta
            
      
    i = 0
    s_old = step_fun(x,*args)
    f = jac(x,*args)
    #or (curvature_condition(f, jac(x + alfa * dx),dx) == False))
    while (
        (armijo_condition(
            s_old, step_fun(x + alfa * dx,*args), alfa, f, dx
        )
        == False and i < 20)):#or (curvature_condition(f,jac(x+alfa*dx,*args), dx)==False) :
        alfa *= beta
        
        i += 1

    return alfa


# @jit(nopython=True,fastmath=True)
def is_pos_def(x):
    """Auxiliary function to test definite positiveness of a matrix"""
    return np.all(np.linalg.eigvals(x) > 0)


# @jit(nopython=True,fastmath=True)
def hessian_regulariser_function(B, eps):
    """Auxiliary function to regularise hessian during Newton Optimisation."""
    
    sh1 = B.shape[0]
    B = (B + B.transpose()) * 0.5  # symmetrization    
    B = B + np.identity(sh1) * eps
    return B

# @jit(nopython=True,fastmath=True)
def solver_newton(x0,ll,jac,hess,args=(),
                        trust_mult = 1e-5,lins = linsearch_fun_CRWCM_separated,
                        tol=1e-8,
                        eps=1e-8,
                        beta=0.5,
                        alfa1=1,
                        max_steps=1000,print_steps=np.inf
                        ):
    
    """Newton Optimisation Solver with optimal linear search and hessian trust region regularisation.
    
    :param x0: initial guess for the Optimiser
    :type x0: np.ndarry

    :param ll: criterion function to be minimised.
    :type ll: object

    :param jac: jacobian of the criterion function to be minimised.
    :type jac: object

    :param hess: hessian of the criterion function to be minimised.
    :type hess: object

    :param args: args of the criterion function.
    :type args: tuple

    :param trust_mult: trust radius for hessian regularisation, default is 1e-05
    :type trust_mult: float

    :param lins: function for optimal linear search, default is the linsearch function for the CRWCM.
    :type lins: object

    :param tol: tolerance for the infinite norm of solution jacobian, default is 1e-08.
    :type tol: float

    :param eps: tolerance for difference of parameters during optimisation, default is 1e-08.
    :type eps: float

    :param beta: multiplier for the step during optimal linear search procedure.
    :type beta: float
    
    :param alfa1: initialisation for the linear search parameter alfa.
    :type alfa1: float
    
    :param max_steps: maximum step during Newton optimisation, default is 1000.
    :type max_steps: float

    :param print_steps: interval in which steps are printed, default is np.inf (no printing).
    :type print_steps: float


    :return: solution parameters
    :rtype: np.ndarray
    
    """

    n_steps = 0
    x = x0  # initial point
    n = len(x0)
    f = jac(x,*args)
    norm = (np.abs(f)).max()
    diff = 1.
    
    while (
        norm > tol and diff > eps and n_steps < max_steps
    ):  # stopping condition
        
        x_old = x  # save previous iteration

                
        H = hess(x,*args)  # original jacobian
        B = hessian_regulariser_function(
                H, np.max(np.abs(jac(x,*args))) *trust_mult
            )
        
        dx = np.linalg.solve(B, -f)

        alfa1 = 1        
        X = (x, dx, beta, alfa1, jac, ll)
        alfa = lins(X,args)
        x = x + alfa * dx        
            
        f = jac(x,*args)

        # stopping condition computation
        norm = (np.abs(f)).max()
        diff = np.abs(x-x_old).max()

        n_steps += 1
        if n_steps % print_steps == 0:
            print('n_steps', n_steps)
            print('norm', norm)
            print('diff', diff)
            
            print('-------------------------------')
                
    return x




def solve_DBCM(k_out,k_in,use_guess=np.array([]),tol=5e-03,maxiter=10):

    """Solver for DBCM.
    
    :param k_out: empirical out-degree
    :type k_out: np.nadarray

    :param k_in: empirical in-degree
    :type k_in: np.nadarray

    :param use_guess: initial guess for the solver.
    :type use_guess: np.nadarray
    
    :param tol: tolerance for the infinite norm of the jacobian for DBCM, default is 5e-03.
    :type tol: float

    :param maxiter: maximum number of iterations for the solver, default is 10.
    :type maxiter: float
    
    :return: solution of the optimisation problem.
    :rtype: np.ndarray

    """
    n_suppliers = len(k_out)
    n_users = len(k_in)
    total_dim = n_suppliers + n_users 
    if len(use_guess) == 0:
        opty = np.random.random(total_dim)
        
    else:
        opty = use_guess
    nrm = np.linalg.norm(jac_DBCM(opty,k_out,k_in),ord=np.inf)
    if nrm < tol:
        return opty  
    for i in range(int(maxiter)):
        # opty = lsq(jac_DBCM,opty,jac=hess_DBCM,args=(k_out,k_in)).x
        opty = lsq(jac_DBCM,opty,args=(k_out,k_in)).x
        # opty = solver_bfgs(x0=opty,ll=ll_DBCM,jac=jac_DBCM,args=(k_out,k_in),lins=linsearch_fun_RBCM)
        # opty = solver_newton(x0=opty,ll=ll_DBCM,jac=jac_DBCM,hess=hess_DBCM,args=(k_out,k_in),
        #                          lins=linsearch_fun_RBCM,max_steps=1000,print_steps=1)
        
        nrm = np.linalg.norm(jac_DBCM(opty,k_out,k_in),ord=np.inf)
        if nrm < tol:
            return opty

    return opty

def solve_RBCM(k_right,k_left,k_rec,use_guess=np.array([]),tol=5e-03,maxiter=10):

    """Solver for RBCM.
    
    :param k_right: empirical non-reciprocated out-degree
    :type k_right: np.nadarray

    :param k_left: empirical non-reciprocated in-degree
    :type k_left: np.nadarray

    :param k_rec: empirical reciprocated degree
    :type k_rec: np.nadarray

    :param use_guess: initial guess for the solver.
    :type use_guess: np.nadarray
    
    :param tol: tolerance for the infinite norm of the jacobian for DBCM, default is 5e-03.
    :type tol: float

    :param maxiter: maximum number of iterations for the solver, default is 10.
    :type maxiter: float
    
    :return: solution of the optimisation problem.
    :rtype: np.ndarray

    """
    
    n_suppliers = len(k_right)
    n_users = len(k_left)
    total_dim = n_suppliers + n_users + n_suppliers
    if len(use_guess) == 0:
        opty = np.random.random(total_dim)
        # opty = np.ones(total_dim)
    else:
        opty = use_guess
    nrm = np.linalg.norm(jac_RBCM(opty,k_right,k_left,k_rec),ord=np.inf)
    if nrm < tol:
        return opty  
    for i in range(int(maxiter)):
        opty = solver_newton(x0=opty,ll=ll_RBCM,jac=jac_RBCM,hess=hess_RBCM,args=(k_right,k_left,k_rec),
                                 lins=linsearch_fun_RBCM,max_steps=1000)
        
        nrm = np.linalg.norm(jac_RBCM(opty,k_right,k_left,k_rec),ord=np.inf)
        if nrm < tol:
            return opty

    return opty

def solve_CReMa_after_DBCM(s_out,s_in,params_DBCM,verbose=0,use_guess=np.array([]),maxiter = 10, tol=1e-07):

    """Solver for CReMa after the solution of the DBCM.
    
    :param s_out: empirical out-strength
    :type s_out: np.nadarray

    :param s_in: empirical in-strength
    :type s_in: np.nadarray

    :param params_DBCM: solution for the DBCM.
    :type params_DBCM: np.nadarray

    :param use_guess: initial guess for the solver.
    :type use_guess: np.ndarray
        
    :param maxiter: maximum number of iterations for the solver, default is 10.
    :type maxiter: float
    
    :param tol: tolerance for the infinite norm of the jacobian for DBCM, default is 5e-03.
    :type tol: float

    :return: solution of the optimisation problem.
    :rtype: np.ndarray

    """
    
    n = len(s_out)
    if len(use_guess) == 0:
        sol = np.random.random(2*n)
        
    else:
        sol = use_guess
        
        
    norm = 10000000
    step = 0

    ll_linspace = []
    sol_linspace = []

    while step < maxiter and norm > tol:
        jac = jac_CReMa_after_DBCM(sol,s_out,s_in,params_DBCM)
        norm_pre = np.linalg.norm(jac,ord=np.inf)   
        relative_error = relative_error_CReMa_after_DBCM(sol,s_out,s_in,params_DBCM)
        norm_rel_pre = np.linalg.norm(relative_error,ord=np.inf)   
            
        
        
        sol = solver_newton(x0=sol,ll=ll_CReMa_after_DBCM,jac=jac_CReMa_after_DBCM,hess=hess_CReMa_after_DBCM,
        args=(s_out,s_in,params_DBCM),max_steps=1000,lins=linsearch_fun_CRWCM_separated)
        
        
        jac = jac_CReMa_after_DBCM(sol,s_out,s_in,params_DBCM)
        relative_error = relative_error_CReMa_after_DBCM(sol,s_out,s_in,params_DBCM)
        ll = ll_CReMa_after_DBCM(sol,s_out,s_in,params_DBCM)
        norm = np.linalg.norm(jac,ord=np.inf)   
        norm_rel = np.linalg.norm(relative_error,ord=np.inf)   
            
        ll_linspace.append(ll)
        sol_linspace.append(sol)
        step += 1

        
        if norm_pre == norm:
            print('blocked at step',step, 'with norm',norm)
            break
    
    return sol


def solve_CRWCM_after_RBCM(s_right,s_left,s_rec_out,s_rec_in,params_RBCM,verbose=0,use_guess=np.array([]),maxiter = 10, tol=1e-07):
    
    """Solver for CRWCM after the solution of the RBCM.
    
    :param s_right: empirical non-reciprocated out-strength
    :type s_right: np.nadarray

    :param s_left: empirical non-reciprocated in-strength
    :type s_left: np.nadarray

    :param s_rec_out: empirical reciprocated out-strength
    :type s_rec_out: np.nadarray

    :param s_rec_in: empirical reciprocated in-strength
    :type s_rec_in: np.nadarray

    :param params_RBCM: solution for the RBCM.
    :type params_RBCM: np.nadarray

    :param use_guess: initial guess for the solver.
    :type use_guess: np.ndarray
        
    :param maxiter: maximum number of iterations for the solver, default is 10.
    :type maxiter: float
    
    :param tol: tolerance for the infinite norm of the jacobian for DBCM, default is 1e-07.
    :type tol: float

    :return: solution of the optimisation problem.
    :rtype: np.ndarray

    """
    
    n = len(s_right)
    if len(use_guess) == 0:
        sol = np.random.random(4*n)
        
    else:
        sol = use_guess
        
        
    norm = 10000000
    step = 0

    ll_linspace = []
    sol_linspace = []

    while step < maxiter and norm > tol:
        jac = jac_CRWCM_after_RBCM(sol,s_right,s_left,s_rec_out,s_rec_in,params_RBCM)
        norm_pre = np.linalg.norm(jac,ord=np.inf)   
        relative_error = relative_error_CRWCM_after_RBCM(sol,s_right,s_left,s_rec_out,s_rec_in,params_RBCM)
        norm_rel_pre = np.linalg.norm(relative_error,ord=np.inf)   
            
        
        
        sol_rl = sol[:2*n]
        sol_rec = sol[2*n:]

    
        sol_rl = solver_newton(x0=sol_rl,ll=ll_CRWCM_after_RBCM_rl,jac=jac_CRWCM_after_RBCM_rl,hess=hess_CRWCM_after_RBCM_rl,
        args=(s_right,s_left,params_RBCM),max_steps=1000,lins=linsearch_fun_CRWCM_separated)
        sol_rec = solver_newton(x0=sol_rec,ll=ll_CRWCM_after_RBCM_rec,jac=jac_CRWCM_after_RBCM_rec,hess=hess_CRWCM_after_RBCM_rec,
        args=(s_rec_out,s_rec_in,params_RBCM),max_steps=1000,lins=linsearch_fun_CRWCM_separated)
        
        sol = np.concatenate((sol_rl,sol_rec))
    
        
        jac = jac_CRWCM_after_RBCM(sol,s_right,s_left,s_rec_out,s_rec_in,params_RBCM)
        relative_error = relative_error_CRWCM_after_RBCM(sol,s_right,s_left,s_rec_out,s_rec_in,params_RBCM)
        ll = ll_CRWCM_after_RBCM(sol,s_right,s_left,s_rec_out,s_rec_in,params_RBCM)
        norm = np.linalg.norm(jac,ord=np.inf)   
        norm_rel = np.linalg.norm(relative_error,ord=np.inf)   
            
        ll_linspace.append(ll)
        sol_linspace.append(sol)
        step += 1

        
        if norm_pre == norm:
            print('blocked at step',step, 'with norm',norm)
            break
    
    return sol





# @jit(nopython=True,fastmath=True)
def matrixate(input):

    """Compute a numpy matrix from corresponding flattened vector"""
    N = int(np.sqrt(len(input)))
    matrix = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            ij = i*N+j
            matrix[i,j] = input[ij]
    return matrix

# @jit(nopython=True,fastmath=True)
def binarize(input):

    """Compute binary projection of numpy matrix"""
    n_suppliers = input.shape[0]
    n_users = input.shape[1]
    byn = np.zeros((n_suppliers,n_users))
    for i in range(n_suppliers):
        for j in range(n_users):   
            if j != i: 
                if input[i,j]>0:
                    byn[i,j] = 1.
    return byn

# @jit(nopython=True,fastmath=True)
def symmetrize(input):

    """Symmetrize numpy matrix in input."""
    n_suppliers = input.shape[0]
    n_users = input.shape[1]
    if n_suppliers == n_users:
        symm = np.zeros((n_suppliers,n_users))
        for i in range(n_suppliers):
            for j in range(i,n_users):   
                if j != i:
                    symm[i,j] = (input[i,j]+input[j,i])/2.
    symm = symm + symm.T
                    
    return symm


# @jit(nopython=True,fastmath=True)
def L_rec(adj):

    """Computes number of reciprocated links."""

    reciprocated_edges = 0
    N = adj.shape[0]
    for i in range(N):
        for j in range(N):
            if j!=i:
                reciprocated_edges += adj[i,j]*adj[j,i]
            
    return reciprocated_edges


# @jit(nopython=True,fastmath=True)
def n_edges_func(adj):

    """Computes number of links."""

    n_suppliers = adj.shape[0]
    n_users = adj.shape[1]
    n_edges = 0
    for i in range(n_suppliers):
        for j in range(n_users):
            if j!=i:
                n_edges += adj[i,j]

    return n_edges

# @jit(nopython=True,fastmath=True)
def gen_binary_adjacencies(adj):

    """Computes reciprocated and non-reciprocated components of binary adjacencies."""

    n = adj.shape[0]
    adj_right = np.empty((n,n))
    adj_left = np.empty((n,n))
    adj_rec = np.empty((n,n))
    adj_unrec = np.empty((n,n))
    
    for i in range(n):
        for j in range(n):
            if j==i:
                adj_right[i,j] = 0.
                adj_left[i,j] = 0.
                adj_rec[i,j] = 0.
                adj_unrec[i,j] = 0.
            else:
                aij = adj[i,j]
                aji = adj[j,i]

                adj_right[i,j] = aij*(1-aji)
                adj_left[i,j] = aji*(1-aij)
                adj_rec[i,j] = aij*aji
                adj_unrec[i,j] = (1-aij)*(1-aji)
                
    return adj_right,adj_left,adj_rec,adj_unrec

# @jit(nopython=True,fastmath=True)
def gen_weighted_adjacencies(adj,w_adj):

    """Computes reciprocated and non-reciprocated components of weighted adjacencies for fast computation of triadic intensities."""
    
    n = adj.shape[0]
    w_adj_right = np.empty((n,n))
    w_adj_left = np.empty((n,n))
    w_adj_rec = np.empty((n,n))
    w_adj_unrec = np.empty((n,n))
    
    for i in range(n):
        for j in range(n):
            if j==i:
                w_adj_right[i,j] = 0.
                w_adj_left[i,j] = 0.
                w_adj_rec[i,j] = 0.
                w_adj_unrec[i,j] = 0.
            else:
                aij = adj[i,j]
                aji = adj[j,i]
                wij = w_adj[i,j]
                wji = w_adj[j,i]

                w_adj_right[i,j] = aij*(1-aji)*wij
                w_adj_left[i,j] = aji*(1-aij)*wji
                w_adj_rec[i,j] = aij*aji*wij*wji
                w_adj_unrec[i,j] = (1-aij)*(1-aji)
                
    return w_adj_right,w_adj_left,w_adj_rec,w_adj_unrec


# @jit(nopython=True,fastmath=True)
def gen_rec_weighted_adjacencies(adj,w_adj):

    """Computes reciprocated components of weighted adjacencies for fast computation of triadic fluxes."""
    
    n = adj.shape[0]
    w_adj_rec_out = np.empty((n,n))
    w_adj_rec_in = np.empty((n,n))
    
    for i in range(n):
        for j in range(n):
            if j==i:
                w_adj_rec_out[i,j] = 0.
                w_adj_rec_in[i,j] = 0.
                
            else:
                aij = adj[i,j]
                aji = adj[j,i]
                wij = w_adj[i,j]
                wji = w_adj[j,i]

                w_adj_rec_out[i,j] = aij*aji*wij
                w_adj_rec_in[i,j] = aij*aji*wji
                
    return w_adj_rec_out,w_adj_rec_in


# @jit(nopython=True,fastmath=True)
def deg(adj):

    """Computes the degree centrality given binary adjacency matrix."""
    n_suppliers = adj.shape[0]
    n_users = adj.shape[1]
    deg = np.zeros(n_suppliers)

    for i in range(n_suppliers):
        for j in range(n_users):
            if j!=i:
                deg[i] += adj[i,j]
    
    return deg
    
# @jit(nopython=True,fastmath=True)
def deg_out(adj):

    """Computes the out-degree centrality given binary adjacency matrix."""
    n_suppliers = adj.shape[0]
    n_users = adj.shape[1]
    deg = np.zeros(n_suppliers)

    for i in range(n_suppliers):
        for j in range(n_users):
            if j!=i:
                deg[i] += adj[i,j]
    
    return deg

# @jit(nopython=True,fastmath=True)
def deg_in(adj): return deg(adj.transpose())

# @jit(nopython=True,fastmath=True)
def deg_right(adj):

    """Computes the non-reciprocated out-degree centrality given binary adjacency matrix."""
    n_suppliers = adj.shape[0]
    n_users = adj.shape[1]
    
    k_right = np.zeros(n_suppliers)
    for i in range(n_suppliers):
        for j in range(n_suppliers):
            if j!=i:
                k_right[i] += adj[i,j]*(1.-adj[j,i])
    return k_right

# @jit(nopython=True,fastmath=True)
def deg_left(adj):
    """Computes the non-reciprocated in-degree centrality given binary adjacency matrix."""
    n_suppliers = adj.shape[0]
    n_users = adj.shape[1]
    
    k_left = np.zeros(n_suppliers)
    for i in range(n_suppliers):
        for j in range(n_users):
            if j!=i:
                k_left[i] += adj[j,i]*(1.-adj[i,j])
    return k_left

# @jit(nopython=True,fastmath=True)
def deg_rec(adj):
    """Computes the reciprocated degree centrality given binary adjacency matrix."""
    n_suppliers = adj.shape[0]
    n_users = adj.shape[1]
    
    k_rec = np.zeros(n_suppliers)
    for i in range(n_suppliers):
        for j in range(n_users):
            if j!=i:
                k_rec[i] += adj[j,i]*adj[i,j]
    return k_rec


# @jit(nopython=True,fastmath=True)
def st_right(adj,w_adj):

    """Computes the non-reciprocated out-strength centrality given binary adjacency matrix and weighted adjacency matrix."""
    n_s = adj.shape[0]
    n_u = adj.shape[1]

    st_right = np.zeros(n_s)
    for i in range(n_s):
        for j in range(n_u):
            if j!=i:
                st_right[i] += adj[i,j]*(1.-adj[j,i])*w_adj[i,j]
    return st_right

# @jit(nopython=True,fastmath=True)
def st_left(adj,w_adj):
    """Computes the non-reciprocated in-strength centrality given binary adjacency matrix and weighted adjacency matrix."""
   

    n_s = adj.shape[0]
    n_u = adj.shape[1]

    st_left = np.zeros(n_s)
    for i in range(n_s):
        for j in range(n_u):
            if j!=i:
                st_left[i] += adj[j,i]*(1.-adj[i,j])*w_adj[j,i]
    return st_left

# @jit(nopython=True,fastmath=True)
def st_rec_out(adj,w_adj):
    """Computes the reciprocated out-strength centrality given binary adjacency matrix and weighted adjacency matrix."""
    n_s = adj.shape[0]
    n_u = adj.shape[1]

    st_rec = np.zeros(n_s)
    for i in range(n_s):
        for j in range(n_u):
            if j!=i:
                st_rec[i] += adj[j,i]*adj[i,j]*w_adj[i,j]
    return st_rec

# @jit(nopython=True,fastmath=True)
def st_rec_in(adj,w_adj):
    """Computes the reciprocated in-strength centrality given binary adjacency matrix and weighted adjacency matrix."""
    n_s = adj.shape[0]
    n_u = adj.shape[1]

    st_rec = np.zeros(n_s)
    for i in range(n_s):
        for j in range(n_u):
            if j!=i:
                st_rec[i] += adj[j,i]*adj[i,j]*w_adj[j,i]
    return st_rec

# @jit(nopython=True,fastmath=True)
def triadic_occurrence(adj_1,adj_2,adj_3):
    
    adj_1 = np.ascontiguousarray(adj_1)
    adj_2 = np.ascontiguousarray(adj_2)
    adj_3 = np.ascontiguousarray(adj_3)
    
    return np.trace(adj_1 @ adj_2 @ adj_3)


        

# @jit(nopython=True,fastmath=True)
def triadic_occurrences(adj_right,adj_left,adj_rec,adj_unrec):
    """Computes the triadic occurrence for the 13 triadic occurrences given the reciprocated and 
    non-reciprocated components of the binary adjacency matrix."""
    Nm = np.empty(13)
    
    multiplicity_2 = 1./2.
    multiplicity_3 = 1./3.
    multiplicity_6 = 1./6.

    Nm[0] = triadic_occurrence(adj_left,adj_right,adj_unrec)*multiplicity_2
    Nm[1] = triadic_occurrence(adj_right,adj_right,adj_unrec)
    Nm[2] = triadic_occurrence(adj_rec,adj_right,adj_unrec)
    Nm[3] = triadic_occurrence(adj_unrec,adj_right,adj_left)*multiplicity_2
    Nm[4] = triadic_occurrence(adj_left,adj_right,adj_left)
    Nm[5] = triadic_occurrence(adj_rec,adj_right,adj_left)*multiplicity_2
    Nm[6] = triadic_occurrence(adj_rec,adj_left,adj_unrec)
    Nm[7] = triadic_occurrence(adj_rec,adj_rec,adj_unrec)*multiplicity_2
    Nm[8] = triadic_occurrence(adj_left,adj_left,adj_left)*multiplicity_3
    Nm[9] = triadic_occurrence(adj_left,adj_rec,adj_left)
    Nm[10] = triadic_occurrence(adj_right,adj_rec,adj_left)*multiplicity_2
    Nm[11] = triadic_occurrence(adj_rec,adj_rec,adj_left)
    Nm[12] = triadic_occurrence(adj_rec,adj_rec,adj_rec)*multiplicity_6
    
    return Nm

# @jit(nopython=True,fastmath=True)
def triadic_intensities(adj_right,adj_left,adj_rec,adj_unrec):

    """Computes the triadic intensity for the 13 triadic occurrences given the reciprocated and 
    non-reciprocated components of the weighted adjacency matrix."""
    
    Im = np.empty(13)    

    adj_right_3 = np.power(adj_right,1./3.)
    adj_left_3 = np.power(adj_left,1./3.)
    adj_rec_3 = np.power(adj_rec,1./3.)

    adj_right_2 = np.power(adj_right,1./2.)
    adj_left_2 = np.power(adj_left,1./2.)
    adj_rec_2 = np.power(adj_rec,1./2.)

    adj_right_4 = np.power(adj_right,1./4.)
    adj_left_4 = np.power(adj_left,1./4.)
    adj_rec_4 = np.power(adj_rec,1./4.)

    #adj_right_5 = np.power(adj_right,1./5.)
    adj_left_5 = np.power(adj_left,1./5.)
    adj_rec_5 = np.power(adj_rec,1./5.)

    adj_rec_6 = np.power(adj_rec,1./6.)

    div_2 = 1./2.
    div_3 = 1./3.
    div_4 = 1./4.
    div_5 = 1./5.
    div_6 = 1./6.
    
    Im[0] = triadic_occurrence(adj_left_2,adj_right_2,adj_unrec)*div_2
    Im[1] = triadic_occurrence(adj_right_2,adj_right_2,adj_unrec)*div_2
    Im[2] = triadic_occurrence(adj_rec_3,adj_right_3,adj_unrec)*div_3
    Im[3] = triadic_occurrence(adj_unrec,adj_right_2,adj_left_2)*div_2 #change from right to left instead of transpose
    Im[4] = triadic_occurrence(adj_left_3,adj_right_3,adj_left_3)*div_3
    Im[5] = triadic_occurrence(adj_rec_4,adj_right_4,adj_left_4)*div_4
    Im[6] = triadic_occurrence(adj_rec_3,adj_left_3,adj_unrec)*div_3
    Im[7] = triadic_occurrence(adj_rec_4,adj_rec_4,adj_unrec)*div_4
    Im[8] = triadic_occurrence(adj_left_3,adj_left_3,adj_left_3)*div_3
    Im[9] = triadic_occurrence(adj_left_4,adj_rec_4,adj_left_4)*div_4
    Im[10] = triadic_occurrence(adj_right_4,adj_rec_4,adj_left_4)*div_4
    Im[11] = triadic_occurrence(adj_rec_5,adj_rec_5,adj_left_5)*div_5
    Im[12] = triadic_occurrence(adj_rec_6,adj_rec_6,adj_rec_6)*div_6
    
    
    return Im



# @jit(nopython=True,fastmath=True)
def triadic_fluxes(adj_right,adj_left,adj_rec,adj_unrec,wadj_right,wadj_left,wadj_rec_out,wadj_rec_in):
    
    """Computes the triadic average flux for the 13 triadic occurrences given the reciprocated and 
    non-reciprocated components of the weighted adjacency matrix."""
    multiplicity_2 = 1./2.
    multiplicity_3 = 1./3.
    multiplicity_6 = 1./6.

    div_2 = 1./2.
    div_3 = 1./3.
    div_4 = 1./4.
    div_5 = 1./5.
    div_6 = 1./6.
    Fm = np.empty(13)
    # adj_rightT = adj_right.transpose()
    # wadj_rightT = wadj_right.transpose()
    
    Fm[0] = (triadic_occurrence(wadj_left,adj_right,adj_unrec)+triadic_occurrence(adj_left,wadj_right,adj_unrec))*div_2*multiplicity_2
    Fm[1] = (triadic_occurrence(wadj_right,adj_right,adj_unrec)+triadic_occurrence(adj_right,wadj_right,adj_unrec))*div_2
    Fm[2] = (triadic_occurrence(wadj_rec_out,adj_right,adj_unrec)+triadic_occurrence(wadj_rec_in,adj_right,adj_unrec)+triadic_occurrence(adj_rec,wadj_right,adj_unrec))*div_3
    Fm[3] = (triadic_occurrence(adj_unrec,wadj_right,adj_left)+triadic_occurrence(adj_unrec,adj_right,wadj_left))*div_2*multiplicity_2
    Fm[4] = (triadic_occurrence(wadj_left,adj_right,adj_left)+triadic_occurrence(adj_left,wadj_right,adj_left)+triadic_occurrence(adj_left,adj_right,wadj_left))*div_3
    Fm[5] = (triadic_occurrence(wadj_rec_out,adj_right,adj_left)+triadic_occurrence(wadj_rec_in,adj_right,adj_left)+triadic_occurrence(adj_rec,wadj_right,adj_left)+triadic_occurrence(adj_rec,adj_right,wadj_left))*div_4*multiplicity_2
    Fm[6] = (triadic_occurrence(wadj_rec_out,adj_left,adj_unrec)+triadic_occurrence(wadj_rec_in,adj_left,adj_unrec)+triadic_occurrence(adj_rec,wadj_left,adj_unrec))*div_3
    Fm[7] = (triadic_occurrence(wadj_rec_out,adj_rec,adj_unrec)+triadic_occurrence(wadj_rec_in,adj_rec,adj_unrec)+triadic_occurrence(adj_rec,wadj_rec_out,adj_unrec)+triadic_occurrence(adj_rec,wadj_rec_in,adj_unrec))*div_4*multiplicity_2
    Fm[8] = (triadic_occurrence(wadj_left,adj_left,adj_left)+triadic_occurrence(adj_left,wadj_left,adj_left)+triadic_occurrence(adj_left,adj_left,wadj_left))*div_3*multiplicity_3
    Fm[9] = (triadic_occurrence(wadj_left,adj_rec,adj_left)+triadic_occurrence(adj_left,wadj_rec_out,adj_left)+triadic_occurrence(adj_left,wadj_rec_in,adj_left)+triadic_occurrence(adj_left,adj_rec,wadj_left))*div_4
    Fm[10] = (triadic_occurrence(wadj_right,adj_rec,adj_left)+triadic_occurrence(adj_right,wadj_rec_out,adj_left)+triadic_occurrence(adj_right,wadj_rec_in,adj_left)+triadic_occurrence(adj_right,adj_rec,wadj_left))*div_4*multiplicity_2
    Fm[11] = (triadic_occurrence(wadj_rec_out,adj_rec,adj_left)+triadic_occurrence(wadj_rec_in,adj_rec,adj_left)+triadic_occurrence(adj_rec,wadj_rec_out,adj_left)+triadic_occurrence(adj_rec,wadj_rec_in,adj_left)+triadic_occurrence(adj_rec,adj_rec,wadj_left))*div_5
    Fm[12] = (triadic_occurrence(wadj_rec_out,adj_rec,adj_rec)+triadic_occurrence(wadj_rec_in,adj_rec,adj_rec)+triadic_occurrence(adj_rec,wadj_rec_out,adj_rec)+triadic_occurrence(adj_rec,wadj_rec_in,adj_rec)+triadic_occurrence(adj_rec,adj_rec,wadj_rec_out)+triadic_occurrence(adj_rec,adj_rec,wadj_rec_in))*div_6*multiplicity_6
    
    
    return Fm


# @jit(nopython=True,fastmath=True)
def compute_zscores(stat_emp,stat_exp,stat_down,stat_up,stat_std):

    """Compute z-scores given expected statistics, empirical statistics and wanted percentiles."""
    zscores_Nm = np.empty(13)
    zscores_down_Nm = np.empty(13)
    zscores_up_Nm = np.empty(13)

    for i in range(13):
        if stat_std[i] != 0.:
            zscores_Nm[i] = (stat_emp[i] - stat_exp[i])/stat_std[i]
            zscores_down_Nm[i] = (stat_down[i] - stat_exp[i])/stat_std[i]
            zscores_up_Nm[i] = (stat_up[i] - stat_exp[i])/stat_std[i]
        else:
            if stat_emp[i] == stat_exp[i]:
                zscores_Nm[i] = 0.
                zscores_down_Nm[i] = 0.
                zscores_up_Nm[i] = 0.

            else:
                zscores_Nm[i] = 100.
                zscores_down_Nm[i] = 0.
                zscores_up_Nm[i] = 0.
                
    return zscores_Nm, zscores_down_Nm, zscores_up_Nm
    



# @jit(nopython=True,fastmath=True,cache=False)
def IT_sampling_Exponential(random_array,cond_wij):
    """Inverse Transform Sampling for the Exponential distribution."""
    x_array = -cond_wij*np.log(1.-random_array)
    return x_array
  

# @jit(nopython=True,fastmath=True)
def gen_top_mat_DBCM(params):

    """Generate a sample of the binary adjacency matrix via DBCM"""

    n_nodes = int(len(params)/2)
    x = np.exp(-params[:n_nodes])
    y = np.exp(-params[n_nodes:2*n_nodes])
    
    aij = np.empty((n_nodes,n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                aij[i,j] == 0.
            else:
            
                uniform = np.random.random()
                
                aux_ij = x[i]*y[j]
                
                pij = aux_ij/(1.+aux_ij)
                
                if uniform <= pij:
                    aij[i,j] = 1.
                else:
                    aij[i,j] = 0.

    return aij


# @jit(nopython=True,fastmath=True)
def gen_top_mat_RBCM(params):

    """Generate a sample of the binary adjacency matrix via RBCM"""
    
    
    n_nodes = int(len(params)/3)
    x = np.exp(-params[:n_nodes])
    y = np.exp(-params[n_nodes:2*n_nodes])
    z = np.exp(-params[2*n_nodes:3*n_nodes])

    aij = np.empty((n_nodes,n_nodes))
    for i in range(n_nodes):
        for j in range(i,n_nodes):
            if i == j:
                aij[i,j] == 0.
            else:
            
                uniform = np.random.random()
                
                aux_ij = x[i]*y[j]
                aux_ji = x[j]*y[i]
                aux_und = z[i]*z[j]
                aux = aux_ij + aux_ji + aux_und

                pij_right = aux_ij/(1.+aux)
                pij_left = aux_ji/(1.+aux)
                pij_rec = aux_und/(1.+aux) 
                pij_unrec = 1./(1.+aux)

                    
                aux_right = pij_right
                aux_left = pij_right + pij_left
                aux_rec = pij_right + pij_left + pij_rec

                if uniform <= aux_right:
                    aij[i,j] = 1.
                    aij[j,i] = 0.
                elif uniform > aux_right and uniform <= aux_left:
                    aij[i,j] = 0.
                    aij[j,i] = 1.
                elif uniform > aux_left and uniform <= aux_rec:
                    aij[i,j] = 1.
                    aij[j,i] = 1.
                else:
                    aij[i,j] = 0.
                    aij[j,i] = 0.

    return aij
    


# @jit(nopython=True,fastmath=True)
def gen_w_mat_CReMa(weighted_params,fij_adj):

    """Generate a sample of the weighted adjacency matrix via CReMa"""
    
    
    N = int(len(weighted_params)/2)
    w_mat = np.empty((N,N))
    b_out = weighted_params[:N]
    b_in = weighted_params[N:2*N]
    
        
    for i in range(N):
        for j in range(N):
            if i == j:
                w_mat[i,j] = 0
                    
            else:
                
                aij = fij_adj[i,j]
                
                random_mat = np.random.random()
                
                
                if aij == 1:
                    cond_wij_hat = 1./(b_out[i]+b_in[j])
                    w_mat[i,j] = IT_sampling_Exponential(random_mat,cond_wij_hat)

                else:

                    w_mat[i,j] = 0.


    return w_mat


# @jit(nopython=True,fastmath=True)
def gen_w_mat_CRWCM(weighted_params,fij_adj):

    """Generate a sample of the weighted adjacency matrix via CRWCM"""
    
    N = int(len(weighted_params)/4)
    w_mat = np.empty((N,N))
    b_right = weighted_params[:N]
    b_left = weighted_params[N:2*N]
    b_rec_out = weighted_params[2*N:3*N]
    b_rec_in = weighted_params[3*N:]
    
        
    for i in range(N):
        for j in range(i,N):
            if i == j:
                w_mat[i,j] = 0
                    
            else:
                
                aij = fij_adj[i,j]
                aji = fij_adj[j,i]
                
                aij_right = aij*(1.-aji)
                aij_left = aji*(1.-aij)
                aij_rec = aij*aji
                aij_unrec = (1.-aij)*(1.-aji)

                random_mat = np.random.random()
                
                
                if aij_right == 1:
                    cond_wij_hat = 1./(b_right[i]+b_left[j])

                    w_mat[i,j] = IT_sampling_Exponential(random_mat,cond_wij_hat)
                    w_mat[j,i] = 0.
                    
                elif aij_left == 1:
                    cond_wji_hat = 1./(b_right[j]+b_left[i])
                    w_mat[j,i] = IT_sampling_Exponential(random_mat,cond_wji_hat)
                    w_mat[i,j] = 0.
                    
                elif aij_rec == 1:
                    cond_wij_hat = 1./(b_rec_out[i]+b_rec_in[j])
                    cond_wji_hat = 1./(b_rec_out[j]+b_rec_in[i])
                    w_mat[i,j] = IT_sampling_Exponential(random_mat,cond_wij_hat)
                    w_mat[j,i] = IT_sampling_Exponential(random_mat,cond_wji_hat)
                    
                elif aij_unrec == 1:
                    
                    w_mat[i,j] = 0.
                    w_mat[j,i] = 0.

    return w_mat



jit(nopython=True,fastmath=True,parallel=True)
def occurrence_ensembler_DBCM(params,n_ensemble = 1000,percentiles=(0,100)):
    
    """Computes triadic occurrences as expected according to the DBCM."""
    array_motifs = np.empty((n_ensemble,13))
    single_motifs = np.empty(13)

    for step in prange(n_ensemble):
        aij = gen_top_mat_DBCM(params)
        
        aij_right,aij_left,aij_rec,aij_unrec = gen_binary_adjacencies(aij)
        single_motifs = triadic_occurrences(aij_right,aij_left,aij_rec,aij_unrec)
        array_motifs[step,:] = single_motifs
        
    array_motifs = array_motifs.transpose()
    netstats_motifs = np.empty((4,13))    
    for i in range(13):
        netstats_motifs[0,i] = array_motifs[i,:].mean()
        netstats_motifs[1,i] = array_motifs[i,:].std()
        netstats_motifs[2,i] = np.percentile(array_motifs[i,:],percentiles[0])
        netstats_motifs[3,i] = np.percentile(array_motifs[i,:],percentiles[1])
    
    return netstats_motifs



# @jit(nopython=True,fastmath=True,parallel=True)
def occurrence_ensembler_RBCM(params,n_ensemble = 1000,percentiles=(0,100)):
    
    """Computes triadic occurrences as expected according to the RBCM."""
    
    array_motifs = np.empty((n_ensemble,13))
    single_motifs = np.empty(13)

    for step in prange(n_ensemble):
        aij = gen_top_mat_RBCM(params)
        
        aij_right,aij_left,aij_rec,aij_unrec = gen_binary_adjacencies(aij)
        single_motifs = triadic_occurrences(aij_right,aij_left,aij_rec,aij_unrec)
        array_motifs[step,:] = single_motifs
        
    array_motifs = array_motifs.transpose()
    netstats_motifs = np.empty((4,13))    
    for i in range(13):
        netstats_motifs[0,i] = array_motifs[i,:].mean()
        netstats_motifs[1,i] = array_motifs[i,:].std()
        netstats_motifs[2,i] = np.percentile(array_motifs[i,:],percentiles[0])
        netstats_motifs[3,i] = np.percentile(array_motifs[i,:],percentiles[1])
    
    return netstats_motifs


def occurrence_intensity_fluxes_ensembler_DBCM_CReMa(params,n_ensemble = 1000,percentiles=(0,100)):

    """Computes triadic intensities and flusxes as expected according to the DBCM+CReMa."""
    
    n = int(len(params)/4)
    array_Nm = np.empty((n_ensemble,13))
    array_Im = np.empty((n_ensemble,13))
    array_Fm = np.empty((n_ensemble,13))
    
    for step in range(n_ensemble):
        aij = gen_top_mat_DBCM(params[:2*n])
        wij = gen_w_mat_CReMa(params[2*n:],aij)
        
        aij_right,aij_left,aij_rec,aij_unrec = gen_binary_adjacencies(aij)
        wij_right,wij_left,wij_rec,wij_unrec = gen_weighted_adjacencies(aij,wij)
        wij_rec_out, wij_rec_in = gen_rec_weighted_adjacencies(aij,wij)
        Nm = triadic_occurrences(aij_right,aij_left,aij_rec,aij_unrec)
        Im = triadic_intensities(wij_right,wij_left,wij_rec,wij_unrec)
        Fm = triadic_fluxes(aij_right,aij_left,aij_rec,aij_unrec,wij_right,wij_left,wij_rec_out,wij_rec_in)
        
        array_Nm[step,:] = Nm
        array_Im[step,:] = Im
        array_Fm[step,:] = Fm
        
    array_Nm = array_Nm.transpose()
    array_Im = array_Im.transpose()
    array_Fm = array_Fm.transpose()
    
    
    netstats_Nm = np.empty((4,13))
    netstats_Im = np.empty((4,13))    
    netstats_Fm = np.empty((4,13))    
    
    for i in range(13):
        netstats_Nm[0,i] = array_Nm[i,:].mean()
        netstats_Nm[1,i] = array_Nm[i,:].std()
        netstats_Nm[2,i] = np.percentile(array_Nm[i,:],percentiles[0])
        netstats_Nm[3,i] = np.percentile(array_Nm[i,:],percentiles[1])
        
        netstats_Im[0,i] = array_Im[i,:].mean()
        netstats_Im[1,i] = array_Im[i,:].std()
        netstats_Im[2,i] = np.percentile(array_Im[i,:],percentiles[0])
        netstats_Im[3,i] = np.percentile(array_Im[i,:],percentiles[1])

        netstats_Fm[0,i] = array_Fm[i,:].mean()
        netstats_Fm[1,i] = array_Fm[i,:].std()
        netstats_Fm[2,i] = np.percentile(array_Fm[i,:],percentiles[0])
        netstats_Fm[3,i] = np.percentile(array_Fm[i,:],percentiles[1])

    
    return netstats_Nm,netstats_Im,netstats_Fm

## @jit(nopython=True,fastmath=True)
def occurrence_intensity_fluxes_ensembler_RBCM_CRWCM(params,n_ensemble = 1000,percentiles=(0,100)):

    """Computes triadic intensities and flusxes as expected according to the RBCM+CRWCM."""
    
    n = int(len(params)/7)
    array_Nm = np.empty((n_ensemble,13))
    array_Im = np.empty((n_ensemble,13))
    array_Fm = np.empty((n_ensemble,13))
    
    for step in range(n_ensemble):
        aij = gen_top_mat_RBCM(params[:3*n])
        wij = gen_w_mat_CRWCM(params[3*n:],aij)
        
        aij_right,aij_left,aij_rec,aij_unrec = gen_binary_adjacencies(aij)
        wij_right,wij_left,wij_rec,wij_unrec = gen_weighted_adjacencies(aij,wij)
        wij_rec_out, wij_rec_in = gen_rec_weighted_adjacencies(aij,wij)
        Nm = triadic_occurrences(aij_right,aij_left,aij_rec,aij_unrec)
        Im = triadic_intensities(wij_right,wij_left,wij_rec,wij_unrec)
        Fm = triadic_fluxes(aij_right,aij_left,aij_rec,aij_unrec,wij_right,wij_left,wij_rec_out,wij_rec_in)
        
        array_Nm[step,:] = Nm
        array_Im[step,:] = Im
        array_Fm[step,:] = Fm
        
    array_Nm = array_Nm.transpose()
    array_Im = array_Im.transpose()
    array_Fm = array_Fm.transpose()
    
    
    netstats_Nm = np.empty((4,13))
    netstats_Im = np.empty((4,13))    
    netstats_Fm = np.empty((4,13))    
    
    for i in range(13):
        netstats_Nm[0,i] = array_Nm[i,:].mean()
        netstats_Nm[1,i] = array_Nm[i,:].std()
        netstats_Nm[2,i] = np.percentile(array_Nm[i,:],percentiles[0])
        netstats_Nm[3,i] = np.percentile(array_Nm[i,:],percentiles[1])
        
        netstats_Im[0,i] = array_Im[i,:].mean()
        netstats_Im[1,i] = array_Im[i,:].std()
        netstats_Im[2,i] = np.percentile(array_Im[i,:],percentiles[0])
        netstats_Im[3,i] = np.percentile(array_Im[i,:],percentiles[1])

        netstats_Fm[0,i] = array_Fm[i,:].mean()
        netstats_Fm[1,i] = array_Fm[i,:].std()
        netstats_Fm[2,i] = np.percentile(array_Fm[i,:],percentiles[0])
        netstats_Fm[3,i] = np.percentile(array_Fm[i,:],percentiles[1])

    
    return netstats_Nm,netstats_Im,netstats_Fm
