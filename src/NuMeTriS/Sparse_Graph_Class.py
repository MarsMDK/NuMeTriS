import numpy as np
from . import Utility_Functions_sparse as ut
import matplotlib.pyplot as plt
import scipy.sparse

class Sparse_Graph:
    
    """Graph instance must be initialised with the weighted adjacency matrix in 2D numpy array format.
    On initialization it computes in-degrees, out-degrees, reciprocated degrees, out-strengths, in-strengths, reciprocated strengths and
    triadic statistics such as occurrences, intensities and fluxes.

    :param adjacency: Weighted adjacency matrix in 2D numpy array format.
    :type adjacency:  np.ndarray 
    
    """

    def __init__(
        self,
        adjacency = None
    ):
        if adjacency is None:
            raise ValueError('Adjacency matrix is missing!')
        elif type(adjacency) != scipy.sparse.csr.csr_matrix:
            raise ValueError('Adjacency matrix must be in scipy.sparse.csr.csr_matrix format!')
        
        self.adjacency = adjacency
        self.binary_adjacency = ut.binarize(adjacency)
        self.dseq_right = ut.deg_right(self.binary_adjacency)
        self.dseq_left = ut.deg_left(self.binary_adjacency)
        self.dseq_rec = ut.deg_rec(self.binary_adjacency)
        self.dseq_out = ut.deg_out(self.binary_adjacency)
        self.dseq_in = ut.deg_in(self.binary_adjacency)
        
        self.stseq_right = ut.st_right(self.binary_adjacency,self.adjacency)
        self.stseq_left = ut.st_left(self.binary_adjacency,self.adjacency)
        self.stseq_rec_out = ut.st_rec_out(self.binary_adjacency,self.adjacency)
        self.stseq_rec_in = ut.st_rec_in(self.binary_adjacency,self.adjacency)
        self.stseq_out = ut.deg_out(self.adjacency)
        self.stseq_in = ut.deg_in(self.adjacency)

        self.adj_right_emp,self.adj_left_emp,self.adj_rec_emp,self.adj_unrec_emp = ut.gen_binary_adjacencies(self.binary_adjacency)
        self.w_adj_right_emp,self.w_adj_left_emp,self.w_adj_rec_emp,self.w_adj_unrec_emp = ut.gen_weighted_adjacencies(self.binary_adjacency,self.adjacency)
        self.w_adj_rec_out_emp, self.w_adj_rec_in_emp = ut.gen_rec_weighted_adjacencies(self.binary_adjacency,self.adjacency)
        
        self.Nm_emp =ut.triadic_occurrences(self.adj_right_emp,self.adj_left_emp,self.adj_rec_emp,self.adj_unrec_emp)
        self.Im_emp =ut.triadic_intensities(self.w_adj_right_emp,self.w_adj_left_emp,self.w_adj_rec_emp,self.w_adj_unrec_emp)
        self.Fm_emp = ut.triadic_fluxes(self.adj_right_emp,self.adj_left_emp,self.adj_rec_emp,self.adj_unrec_emp,self.w_adj_right_emp,self.w_adj_left_emp,self.w_adj_rec_out_emp,self.w_adj_rec_in_emp)
        
        self.n_edges = ut.n_edges_func(self.binary_adjacency)
        self.n_suppliers = len(self.dseq_out)
        self.n_nodes = len(self.dseq_out)
        self.n_users = len(self.dseq_in)
        self.L_rec = ut.L_rec(self.binary_adjacency)
        self.L_max = self.n_suppliers*(self.n_suppliers-1)
        self.n_observations = self.n_suppliers*self.n_users
        
        self.implemented_models = ['DBCM','RBCM','DBCM+CReMa','RBCM+CRWCM']
        self.model_binary_adjacency = None
        self.model_weighted_adjacency = None

        self.ll = None
        self.ll_binary = None
        self.jacobian = None
        self.norm = None
        self.aic = None
        self.aic_binary = None
        
        self.args = None
        self.norm_rel = None

        
        self.avg_motif_occurrence_array = None
        self.std_motif_occurrence_array = None
        self.percentiles_inf_motif_occurrence_array = None
        self.percentiles_sup_motif_occurrence_array = None
        
        self.avg_motif_intensity_array = None
        self.std_motif_intensity_array = None
        self.percentiles_inf_motif_intensity_array = None
        self.percentiles_sup_motif_intensity_array = None
        
        
        self.fij_matrix = None
        
    def solver(
        self,
        model,
        imported_params = np.array([]),
        imported_top_params = np.array([]),
        use_guess = np.array([]),
        maxiter = 30,
        verbose=0.,
        tol = 1e-06
        ):

        """Optimize chosen model for Graph instance and compute log-likelihoods, jacobian, infinite norms and relative norms. 
        The available models are DBCM and RBCM for the binary optimisation and DBCM+CReMa and RBCM+CRWCM for the mixture models.
         
        :param model: Chosen model
        :type model: string
        
        :param imported_params: If used, uses wanted parameters as solution, default is empty array.
        :type imported_params: np.ndarray 
        
        :param imported_top_params: If used for mixture models, uses wanted parameters as solution for the binary problem, default is empty array.
        :type imported_params: np.ndarray 
        
        :param use_guess: If used, uses wanted parameters as starters in the optimization, default is empty array.
        :type use_guess: np.ndarray 
        
        :param maxiter: Maximum Iterations of solver function, default is 30.
        :type maxiter: int
        
        :param verbose: True if you want to see every n*print_steps iterations, default is 0.
        :type verbose: boolean
        
        
        :param tol: tolerance for infinite norm in the optimization process, default is 1e-06.
        :type tol: float 
       
        """




        self.model = model
        if model == 'DBCM':
            self.args = (self.dseq_out,self.dseq_in)
        elif model == 'RBCM':
            self.args = (self.dseq_right,self.dseq_left,self.dseq_rec)

        elif model == 'DBCM+CReMa':
            self.args_top = (self.dseq_out,self.dseq_in)
            self.args_weighted = (self.stseq_out,self.stseq_in)
            
        elif model == 'RBCM+CRWCM':
            self.args_top = (self.dseq_right,self.dseq_left,self.dseq_rec)    
            self.args_weighted =(self.stseq_right,self.stseq_left,self.stseq_rec_out,self.stseq_rec_in)
        
        else:
            raise TypeError('Only implemented models are "DBCM", "RBCM", "DBCM+CReMa" and "RBCM+CRWCM"!')

        if model == "DBCM":
            if len(imported_params) == 0:
                self.params = ut.solve_DBCM(*self.args,use_guess=use_guess,maxiter=maxiter,tol=tol)
            else:
                self.params = imported_params
            self.jacobian = ut.jac_DBCM(self.params,*self.args)
            self.norm = np.linalg.norm(self.jacobian,ord=np.inf)
            self.ll = -ut.ll_DBCM(self.params,*self.args)
            
        if model == "RBCM":
            if len(imported_params) == 0:
                self.params = ut.solve_RBCM(*self.args,use_guess=use_guess,maxiter=maxiter,tol=tol)
            else:
                self.params = imported_params
            self.jacobian = ut.jac_RBCM(self.params,*self.args)
            self.norm = np.linalg.norm(self.jacobian,ord=np.inf)
            self.ll = -ut.ll_RBCM(self.params,*self.args)
            

        if model == 'DBCM+CReMa':
            if len(imported_params) == 0 and len(imported_top_params)==0:
                self.topological_params = ut.solve_DBCM(*self.args_top,maxiter=maxiter)
                if len(self.topological_params)!=2*self.n_nodes:
                    raise TypeError('uncorrect dimension for topological params')
                self.weighted_params = ut.solve_CReMa_after_DBCM(*self.args_weighted,self.topological_params,use_guess=use_guess,maxiter=maxiter,tol=tol)
                self.params = np.concatenate((self.topological_params,self.weighted_params))

            elif len(imported_params) == 0 and len(imported_top_params)!=0:
                self.topological_params = imported_top_params
                if len(self.topological_params)!=2*self.n_nodes:
                    raise TypeError('uncorrect dimension for topological params')
                self.weighted_params = ut.solve_CReMa_after_DBCM(*self.args_weighted,self.topological_params,use_guess=use_guess,maxiter=maxiter,tol=tol)
                self.params = np.concatenate((self.topological_params,self.weighted_params))


            else:
                self.params = imported_params
                self.weighted_params = self.params[2*self.n_nodes:]
                self.topological_params = self.params[:2*self.n_nodes]
            
            ll_weighted = -ut.ll_CReMa_after_DBCM(self.weighted_params,*self.args_weighted,self.topological_params)
            ll_topological = -ut.ll_DBCM(self.topological_params,*self.args_top)
            self.ll = ll_topological + ll_weighted


            jac_weighted = -ut.jac_CReMa_after_DBCM(self.weighted_params,*self.args_weighted,self.topological_params)
            jac_topological = -ut.jac_DBCM(self.topological_params,*self.args_top)
            
            relative_error_weighted = -ut.relative_error_CReMa_after_DBCM(self.weighted_params,*self.args_weighted,self.topological_params)
            relative_error_topological = -ut.relative_error_DBCM(self.topological_params,*self.args_top)
            
            self.jacobian = np.concatenate((jac_topological,jac_weighted))
            self.relative_error = np.concatenate((relative_error_topological,relative_error_weighted))
            
            
            self.norm = np.linalg.norm(self.jacobian,ord=np.inf)
            self.norm_top = np.linalg.norm(jac_topological,ord=np.inf)
            self.norm_w = np.linalg.norm(jac_weighted,ord=np.inf)
            self.norm_rel = np.linalg.norm(self.relative_error,ord=np.inf)
            # print('norm rel:',self.norm_rel)          
    
        if model == 'RBCM+CRWCM':
            if len(imported_params) == 0 and len(imported_top_params)==0:
                self.topological_params = ut.solve_RBCM(*self.args_top,maxiter=maxiter,tol=tol)
                if len(self.topological_params)!=3*self.n_nodes:
                    raise TypeError('uncorrect dimension for topological params')
                self.weighted_params = ut.solve_CRWCM_after_RBCM(*self.args_weighted,self.topological_params,use_guess=use_guess,maxiter=maxiter,tol=tol)
                self.params = np.concatenate((self.topological_params,self.weighted_params))

            elif len(imported_params) == 0 and len(imported_top_params)!=0:
                self.topological_params = imported_top_params
                if len(self.topological_params)!=3*self.n_nodes:
                    raise TypeError('uncorrect dimension for topological params')
                self.weighted_params = ut.solve_CRWCM_after_RBCM(*self.args_weighted,self.topological_params,use_guess=use_guess,maxiter=maxiter,tol=tol)
                self.params = np.concatenate((self.topological_params,self.weighted_params))


            else:
                self.params = imported_params
                self.weighted_params = self.params[3*self.n_nodes:]
                self.topological_params = self.params[:3*self.n_nodes]
            
            ll_weighted = -ut.ll_CRWCM_after_RBCM(self.weighted_params,*self.args_weighted,self.topological_params)
            ll_topological = -ut.ll_RBCM(self.topological_params,*self.args_top)
            self.ll = ll_topological + ll_weighted


            jac_weighted = -ut.jac_CRWCM_after_RBCM(self.weighted_params,*self.args_weighted,self.topological_params)
            jac_topological = -ut.jac_RBCM(self.topological_params,*self.args_top)
            
            relative_error_weighted = -ut.relative_error_CRWCM_after_RBCM(self.weighted_params,*self.args_weighted,self.topological_params)
            relative_error_topological = -ut.relative_error_RBCM(self.topological_params,*self.args_top)
            
            self.jacobian = np.concatenate((jac_topological,jac_weighted))
            self.relative_error = np.concatenate((relative_error_topological,relative_error_weighted))
            
            self.norm = np.linalg.norm(self.jacobian,ord=np.inf)
            self.norm_top = np.linalg.norm(jac_topological,ord=np.inf)
            self.norm_w = np.linalg.norm(jac_weighted,ord=np.inf)
            self.norm_rel = np.linalg.norm(self.relative_error,ord=np.inf)
            # print('norm rel:',self.norm_rel)
            
    

    def numerical_triadic_zscores(self,n_ensemble= 1000,percentiles=(2.5,97.5)):

        """Compute triadic network structures on a statistical ensemble of networks. The computed statistics are
        triadic occurrences for DBCM and RBCM, and occurrences, intensities and fluxes for DBCM+CReMa and RBCM+CRWCM.
        After computing triadic statistics, also z-scores and significance scores are computed.
         
        :param n_ensemble: Chosen model, default is 1000.
        :type n_ensemble: float
        
        :param percentiles: Computed percentiles for estimated confidence interval in tuple format, default is (2.5,97.5) correspondent to a 95% CI.
        :type percentiles: tuple
       
        """

        if self.model== 'DBCM':
            res_Nm = ut.occurrence_ensembler_DBCM(self.params, n_ensemble ,percentiles)
            self.avg_Nm = res_Nm[0]
            self.std_Nm = res_Nm[1]
            self.pdown_Nm = res_Nm[2]
            self.pup_Nm = res_Nm[3]


            self.zscores_Nm, self.zscores_down_Nm, self.zscores_up_Nm = ut.compute_zscores(self.Nm_emp,self.avg_Nm,self.pdown_Nm,self.pup_Nm,self.std_Nm)
            norm_zscores_Nm = np.linalg.norm(self.zscores_Nm)
            self.normalized_z_Nm = self.zscores_Nm/norm_zscores_Nm

        
        elif self.model== 'RBCM':
            res_Nm = ut.occurrence_ensembler_RBCM(self.params, n_ensemble ,percentiles)
            self.avg_Nm = res_Nm[0]
            self.std_Nm = res_Nm[1]
            self.pdown_Nm = res_Nm[2]
            self.pup_Nm = res_Nm[3]


            self.zscores_Nm, self.zscores_down_Nm, self.zscores_up_Nm = ut.compute_zscores(self.Nm_emp,self.avg_Nm,self.pdown_Nm,self.pup_Nm,self.std_Nm)
            norm_zscores_Nm = np.linalg.norm(self.zscores_Nm)
            self.normalized_z_Nm = self.zscores_Nm/norm_zscores_Nm

        elif self.model == 'DBCM+CReMa':            

            res_Nm, res_Im, res_Fm = ut.occurrence_intensity_fluxes_ensembler_DBCM_CReMa(self.params, n_ensemble ,percentiles)
            self.avg_Nm = res_Nm[0]
            self.std_Nm = res_Nm[1]
            self.pdown_Nm = res_Nm[2]
            self.pup_Nm = res_Nm[3]

            self.avg_Im = res_Im[0]
            self.std_Im = res_Im[1]
            self.pdown_Im = res_Im[2]
            self.pup_Im = res_Im[3]

            self.avg_Fm = res_Fm[0]
            self.std_Fm = res_Fm[1]
            self.pdown_Fm = res_Fm[2]
            self.pup_Fm = res_Fm[3]

            
            self.zscores_Nm, self.zscores_down_Nm, self.zscores_up_Nm = ut.compute_zscores(self.Nm_emp,self.avg_Nm,self.pdown_Nm,self.pup_Nm,self.std_Nm)
            self.zscores_Im, self.zscores_down_Im, self.zscores_up_Im = ut.compute_zscores(self.Im_emp,self.avg_Im,self.pdown_Im,self.pup_Im,self.std_Im)
            self.zscores_Fm, self.zscores_down_Fm, self.zscores_up_Fm = ut.compute_zscores(self.Fm_emp,self.avg_Fm,self.pdown_Fm,self.pup_Fm,self.std_Fm)
            
            norm_zscores_Nm = np.linalg.norm(self.zscores_Nm)
            norm_zscores_Im = np.linalg.norm(self.zscores_Im)
            norm_zscores_Fm = np.linalg.norm(self.zscores_Fm)
            
            self.normalized_z_Nm = self.zscores_Nm/norm_zscores_Nm
            self.normalized_z_Im = self.zscores_Im/norm_zscores_Im
            self.normalized_z_Fm = self.zscores_Fm/norm_zscores_Fm

        elif self.model == 'RBCM+CRWCM':            

            res_Nm, res_Im, res_Fm = ut.occurrence_intensity_fluxes_ensembler_RBCM_CRWCM(self.params, n_ensemble ,percentiles)
            self.avg_Nm = res_Nm[0]
            self.std_Nm = res_Nm[1]
            self.pdown_Nm = res_Nm[2]
            self.pup_Nm = res_Nm[3]

            self.avg_Im = res_Im[0]
            self.std_Im = res_Im[1]
            self.pdown_Im = res_Im[2]
            self.pup_Im = res_Im[3]

            self.avg_Fm = res_Fm[0]
            self.std_Fm = res_Fm[1]
            self.pdown_Fm = res_Fm[2]
            self.pup_Fm = res_Fm[3]

            
            self.zscores_Nm, self.zscores_down_Nm, self.zscores_up_Nm = ut.compute_zscores(self.Nm_emp,self.avg_Nm,self.pdown_Nm,self.pup_Nm,self.std_Nm)
            self.zscores_Im, self.zscores_down_Im, self.zscores_up_Im = ut.compute_zscores(self.Im_emp,self.avg_Im,self.pdown_Im,self.pup_Im,self.std_Im)
            self.zscores_Fm, self.zscores_down_Fm, self.zscores_up_Fm = ut.compute_zscores(self.Fm_emp,self.avg_Fm,self.pdown_Fm,self.pup_Fm,self.std_Fm)
            
            norm_zscores_Nm = np.linalg.norm(self.zscores_Nm)
            norm_zscores_Im = np.linalg.norm(self.zscores_Im)
            norm_zscores_Fm = np.linalg.norm(self.zscores_Fm)
            
            self.normalized_z_Nm = self.zscores_Nm/norm_zscores_Nm
            self.normalized_z_Im = self.zscores_Im/norm_zscores_Im
            self.normalized_z_Fm = self.zscores_Fm/norm_zscores_Fm
            
        

            
            
    def plot_zscores(self,color = 'blue', linestyle='-',marker='o',alpha = 0.3,export_path_zscores='',
                     export_path_significance='',type='z-scores',label=None,show = False):



        """Plot function for z-scores. If the model is DBCM or RBCM it plots z-scores for triadic occurrences.
        If it is DBCM+CReMa and RBCM+CRWCM, it plots z-scores for triadic occurrences, intensities and fluxes.
         
        :param color: Color of profiles and CIs, default is 'blue'.
        :type color: string
        

        :param linestyle: Linestyle for profiles.
        :type linestyle: string
        
        :param marker: Marker for profiles
        :type marker: string

        :param alpha: Degree of transparency for CI.
        :type alpha: float

        :param export_path_zscores: Export path for z-scores plot.
        :type alpha: string
        
        :param export_path_zscores: Export path for significance plot.
        :type alpha: string

        :param type: Type of plot, 'z-scores', 'significance' or 'both'.
        :type alpha: string

        :param label: Label for the legend in the plot.
        :type label: string
        
        :param show: Arg for showing the plot in display.
        :type show: boolean
        
        """

        if type not in ['z-scores', 'significance', 'both']:
            raise ValueError('only possible type are "z-scores" and "significance" or "both')

        if show not in [False, True]:
            raise ValueError('only possible type are "False" or "True"')

        model = self.model
        if label == None:
            label = model
        m = np.arange(13)+1
        if model in ['DBCM','RBCM']:
            
            if type in ['z-scores','both']:
                fig,ax = plt.subplots()
                plt.suptitle('z-score profile')
                
                plt.plot(m,self.zscores_Nm,color =color,label=label,linestyle=linestyle,marker=marker)
                plt.fill_between(m,self.zscores_down_Nm,self.zscores_up_Nm,color=color,alpha=alpha)
                #ax.legend()
                plt.ylabel('$z_{Nm}$')
                plt.xlabel('m')
                plt.tight_layout()
                if export_path_zscores != '':
                    plt.savefig(export_path_zscores)
                if show == True:
                    plt.show()
                plt.close()

            if type in ['significance','both']:

                fig,ax = plt.subplots()
                plt.suptitle('significance profile')
                
                plt.plot(m,self.normalized_z_Nm,color =color,label=label,linestyle=linestyle,marker=marker)
                #ax.legend()
                plt.ylabel('$rz_{Nm}$')
                plt.xlabel('m')
                plt.tight_layout()
                if export_path_significance != '':
                    plt.savefig(export_path_significance)
                if show == True:
                    plt.show()
                plt.close()
            
            
        elif model in ['DBCM+CReMa', 'RBCM+CRWCM']:

            if type in ['z-scores','both']:
                fig,ax = plt.subplots(3,1,sharex=True)
                plt.suptitle('z-score profile')
                ax[0].plot(m,self.zscores_Nm,color =color,label=label,linestyle=linestyle,marker=marker)
                ax[0].fill_between(m,self.zscores_down_Nm,self.zscores_up_Nm,color=color,alpha=alpha)
                ax[1].plot(m,self.zscores_Im,color =color,label=label,linestyle=linestyle,marker=marker)
                ax[1].fill_between(m,self.zscores_down_Im,self.zscores_up_Im,color=color,alpha=alpha)
                ax[2].plot(m,self.zscores_Fm,color =color,label=label,linestyle=linestyle,marker=marker)
                ax[2].fill_between(m,self.zscores_down_Fm,self.zscores_up_Fm,color=color,alpha=alpha)
                
                #ax.legend()
                ax[0].axhline(y=1.96,linestyle='--',color='black')
                ax[0].axhline(y=-1.96,linestyle='--',color='black')
                ax[1].axhline(y=1.96,linestyle='--',color='black')
                ax[1].axhline(y=-1.96,linestyle='--',color='black')
                ax[2].axhline(y=1.96,linestyle='--',color='black')
                ax[2].axhline(y=-1.96,linestyle='--',color='black')
                
                ax[0].set_ylabel('$z_{Nm}$',fontsize=16)
                ax[2].set_ylabel('$z_{Fm}$',fontsize=16)
                ax[1].set_ylabel('$z_{Im}$',fontsize=16)
                
                ax[2].set_xlabel('m',fontsize=16)
                
                plt.tight_layout()
                if export_path_zscores != '':
                    plt.savefig(export_path_zscores)
                if show == True:
                    plt.show()
                plt.close()

            if type in ['significance','both']:
                fig,ax = plt.subplots(3,1,sharex=True)
                plt.suptitle('significance profile')
                ax[0].plot(m,self.normalized_z_Nm,color =color,label=label,linestyle=linestyle,marker=marker)
                ax[1].plot(m,self.normalized_z_Im,color =color,label=label,linestyle=linestyle,marker=marker)
                ax[2].plot(m,self.normalized_z_Fm,color =color,label=label,linestyle=linestyle,marker=marker)
                
                #ax.legend()
                
                ax[0].set_ylabel('$rz_{Nm}$',fontsize=16)
                ax[2].set_ylabel('$rz_{Fm}$',fontsize=16)
                ax[1].set_ylabel('$rz_{Im}$',fontsize=16)
                
                ax[2].set_xlabel('m',fontsize=16)
                
                plt.tight_layout()
                if export_path_significance != '':
                    plt.savefig(export_path_significance)
                if show == True:
                    plt.show()
                plt.close()

        
        

    

                