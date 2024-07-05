from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def gaussian_expected_top_r_of_n(r, n, mu, sigma, method='approx'):
    '''
    A function to calculate the order statistics of Gaussian
    Two methods implemented, monte-carlo and approximation
    '''
    if np.min(r)<1:
        return 0
 
    if method == 'monte-carlo':
        Nsamples = 1000
        val = 0
        for i in range(Nsamples):
            scores = mu + sigma*np.random.randn(n,1)
            sorted_scores = np.sort(scores.flatten())
            sorted_scores = sorted_scores[::-1]
            val = val + sorted_scores[r-1]
        val = val/Nsamples
    
    elif method == 'approx':
        alpha = 0.375
        p = (r - alpha)/(n-2*alpha+1)
        val = -norm.ppf(p)
        val = sigma*val+mu
    
    return val

def get_greedy_reward(scores_u, scores_v, N_t, N_u, c_k, action_k, remnant_app_u, remnant_app_total):
    '''
    A function to calculate the greedy reward of the institution K_idx
    '''
    remnant_app_v = remnant_app_total - remnant_app_u
    A_k= np.round(c_k*N_t)
    A_u = action_k*A_k
    A_v = A_k - A_u
    if remnant_app_u - A_u < 0 or remnant_app_v - A_v < 0:
        greedy_reward = -np.Inf
    else:
        greedy_reward = 0
        reward_u = 0
        reward_v = 0

        N_v = N_t - N_u
        offset_u = N_u - remnant_app_u
        offset_v = N_v - remnant_app_v

        reward_u = np.sum(scores_u[int(offset_u):int(offset_u+A_u)])
        reward_v = np.sum(scores_v[int(offset_v):int(offset_v+A_v)])

        greedy_reward = (1/A_k)*(reward_u + reward_v)
        
    return greedy_reward

def process_MFG(N, c, alpha, mean, std, method, theta_init, lambda_, eta_init, num_instances=1, num_rounds=200, decay=0, 
                    decay_steps=100, smallest_eta=1e-3, epsilon=0.01, reinf_method='positive', role_model_ratio=1, weights=[], order=1):
    '''
    A function to process MFG policy, returns averages over # of instances 
    '''

    K = len(c)

    # defining vectors to keep record of the applicants, admissions ratio and the mean parameter theta
    applicants_u = np.zeros([K,num_rounds])
    admitted_u = np.zeros([K,num_rounds])
    applicants_u_avg = np.zeros([K,num_rounds])
    admitted_u_avg = np.zeros([K,num_rounds])


    theta_vec = np.zeros(num_rounds+1)
    theta_vec_avg = np.zeros(num_rounds+1)
    theta_vec[0] = theta_init    
    
    role_model = np.zeros([K,num_rounds])
    role_model_avg = np.zeros([K,num_rounds])

    if not isinstance(lambda_, np.ndarray):
        lambda_ = np.ones(K)*lambda_

    # iteration over instances
    for iter in tqdm(range(num_instances)):
        theta = theta_init
        eta = eta_init
        # iteration over rounds
        for i in range(num_rounds):
            # decaying eta parameter
            if (i+1)%decay_steps == 0 and decay == 1 and eta > smallest_eta:
                eta = eta*0.75
            # sampling state and setting the initial values of the states
            N_u = np.random.poisson(theta*N)
            N_v = np.random.poisson((1-theta)*N)
            N_t = N_u + N_v
            N_u = np.round(N*(N_u/N_t))
            N_v = N - N_u
            N_t = N
            if N_u == 0:
                scores_u = []
            else:
                scores_u = gaussian_expected_top_r_of_n(np.arange(1,N_u+1), N_u, mean[0], std[0], method)
            if N_v == 0:
                scores_v = []
            else:
                scores_v = gaussian_expected_top_r_of_n(np.arange(1,N_v+1), N_v, mean[1], std[1], method)
            A_k = np.round(c*N_t)
            remnant_app_u = N_u
            remnant_app_v = N_v
            remnant_app_total = N_t
            optimal_actions = np.zeros(K)
            # role model
            role_model_u = np.zeros(K)
            # iteration over institutions, first the best institution picks, then the following one picks and go on
            for k in range(K):
                # make a list of actions for each institution
                actions = np.arange(A_k[k]+1)/A_k[k]
                greedy_reward_k = np.zeros(len(actions))
                applicants_u[k][i] = remnant_app_u/remnant_app_total
                # reward for each action for institution k
                for j in range(len(actions)):
                    greedy_reward_k[j] = get_greedy_reward(scores_u, scores_v, N_t, N_u, c[k], actions[j], remnant_app_u, remnant_app_total)
                # picks the best action and updates the remaining people pools
                overall_reward_k = greedy_reward_k - lambda_[k]*np.square(actions - alpha)
                optimal_action_idx = np.argmax(overall_reward_k)
                optimal_actions[k] = actions[optimal_action_idx]
                admitted_u[k][i] = optimal_actions[k]
                accepted_app_u = A_k[k]*optimal_actions[k]
                

                if reinf_method=='role_model':
                    selected_scores_u = scores_u[int(N_u-remnant_app_u):int(N_u-remnant_app_u+accepted_app_u)]
                    selected_scores_v = scores_v[int(N_v-remnant_app_v):int(N_v-remnant_app_v+A_k[k]-accepted_app_u)]
                    selected_applicant_scores = np.append(selected_scores_u, selected_scores_v)
                    kth = int(role_model_ratio*len(selected_applicant_scores))
                    cut_off_role_model_score = np.sort(selected_applicant_scores)[::-1][kth-1]
                    # Fraction of group u among role model candidates for institution k 
                    role_model_u[k] = np.sum(selected_scores_u >=cut_off_role_model_score) \
                            /(role_model_ratio*A_k[k])
                    role_model[k][i] = role_model_u[k]

                remnant_app_total = remnant_app_total - A_k[k]
                remnant_app_u = remnant_app_u - accepted_app_u
                remnant_app_v = remnant_app_v - (A_k[k] - accepted_app_u)

            # calculates the weighted average of actions
            if reinf_method=='positive':
                opt_act_weighted = np.dot(optimal_actions,c)/np.sum(c)
                update = (opt_act_weighted - N_u/N_t)
            elif reinf_method=='weight':
                opt_act_weighted = np.dot(optimal_actions,weights*c)/np.sum(weights*c)
                update = (opt_act_weighted - N_u/N_t)
            elif (reinf_method=='role_model'):
                opt_act_weighted = np.dot(role_model_u, c)/np.sum(c)
                update = (opt_act_weighted - N_u/N_t)
            elif reinf_method=='indiv_pure_pos_reinf':
                update = np.dot((optimal_actions-N_u/N_t), weights)/np.sum(weights)
            elif reinf_method=='order':
                opt_act_weighted = np.dot(optimal_actions,c)/np.sum(c)
                update = (opt_act_weighted - N_u/N_t)
                update = np.sign(update)*np.power(np.abs(update),order)
            else:
                raise NotImplementedError

            # updates the mean parameter
            theta = min(max(epsilon, theta+eta*update),1-epsilon)
            theta_vec[i+1] = theta       

        # cumulates the results from the instances
        applicants_u_avg += applicants_u
        admitted_u_avg += admitted_u
        theta_vec_avg += theta_vec
        role_model_avg += role_model

    # calculates the average over the instances
    print('MFG Setting, all instances are finished.')
    applicants_u_avg = applicants_u_avg/num_instances
    admitted_u_avg = admitted_u_avg/num_instances
    theta_vec_avg = theta_vec_avg/num_instances   
    role_model_avg = role_model_avg/num_instances 

    return applicants_u_avg, admitted_u_avg, theta_vec_avg, role_model_avg

def process_CMFG(N, c, alpha, mean, std, method, theta_init, lambda_, eta_init, num_instances=1, num_rounds=200, decay=0, 
                    decay_steps=100, smallest_eta=1e-3, epsilon=0.01, reinf_method='positive', role_model_ratio=1, weights=[], order=1):
    '''
    A function to process CMFG policy, returns averages over # of instances 
    '''

    K = len(c)

    # defining vectors to keep record of the applicants, admissions ratio and the mean parameter theta
    applicants_u = np.zeros([K,num_rounds])
    admitted_u = np.zeros([K,num_rounds])
    applicants_u_avg = np.zeros([K,num_rounds])
    admitted_u_avg = np.zeros([K,num_rounds])


    theta_vec = np.zeros(num_rounds+1)
    theta_vec_avg = np.zeros(num_rounds+1)
    theta_vec[0] = theta_init    
    
    role_model = np.zeros([K,num_rounds])
    role_model_avg = np.zeros([K,num_rounds])

    # save optimal actions for each round
    optimal_actions_history = {}

    if not isinstance(lambda_, np.ndarray):
        lambda_ = np.ones(K)*lambda_

    # iteration over instances
    for iter in tqdm(range(num_instances)):
        theta = theta_init
        eta = eta_init
        # iteration over rounds
        for i in range(num_rounds):
            # decaying eta parameter
            if (i+1)%decay_steps == 0 and decay == 1 and eta > smallest_eta:
                eta = eta*0.75
            # sampling state and setting the initial values of the states
            N_u = np.random.poisson(theta*N)
            N_v = np.random.poisson((1-theta)*N)
            N_t = N_u + N_v
            N_u = np.round(N*(N_u/N_t))
            N_v = N - N_u
            N_t = N
            if N_u == 0:
                scores_u = []
            else:
                scores_u = gaussian_expected_top_r_of_n(np.arange(1,N_u+1), N_u, mean[0], std[0], method)
            if N_v == 0:
                scores_v = []
            else:
                scores_v = gaussian_expected_top_r_of_n(np.arange(1,N_v+1), N_v, mean[1], std[1], method)
            A_k = np.round(c*N_t)
            key = str(N_u) + '_' + str(N_v)
            if key in optimal_actions_history:
                optimal_actions = optimal_actions_history[key]
            else:
                min_acc_u = np.maximum(0, np.sum(A_k)-N_v)
                max_acc_u = np.minimum(np.sum(A_k), N_u)

                # Preallocate memory for actions
                actions = []

                for acc_u in range(int(min_acc_u), int(max_acc_u)+1):
                    # Use list comprehension instead of vstack for faster operations
                    actions.extend([i for i in np.argwhere(np.sum(np.array(np.meshgrid(*[np.arange(acc_u+1)]*K)), axis=0) == acc_u) if all(i <= A_k)])

                # Convert actions to numpy array after all actions have been added
                actions = np.array(actions)
    
                reward = np.zeros(actions.shape[0])

                # Use enumerate for cleaner code
                for j, action in enumerate(actions):
                    action = action/A_k
                    remnant_app_u = N_u
                    remnant_app_v = N_v
                    remnant_app_total = N_t
                    for k in range(K):
                        reward[j] += get_greedy_reward(scores_u, scores_v, N_t, N_u, c[k], action[k], remnant_app_u, remnant_app_total)
                        reward[j] -= lambda_[k]*np.square(action[k] - alpha)
                        remnant_app_total -= A_k[k]
                        remnant_app_u -= A_k[k]*action[k]

                optimal_action_idx = np.argmax(reward)
                optimal_actions = actions[optimal_action_idx]/A_k
                optimal_actions_history[key] = optimal_actions
            # updates the remaining people pools
            remnant_app_u = N_u
            remnant_app_v = N_v
            remnant_app_total = N_t
            role_model_u = np.zeros(K)
            for k in range(K):
                applicants_u[k][i] = remnant_app_u/remnant_app_total
                admitted_u[k][i] = optimal_actions[k]
                accepted_app_u = A_k[k]*optimal_actions[k]

                if reinf_method=='role_model':
                    selected_scores_u = scores_u[int(N_u-remnant_app_u):int(N_u-remnant_app_u+accepted_app_u)]
                    selected_scores_v = scores_v[int(N_v-remnant_app_v):int(N_v-remnant_app_v+A_k[k]-accepted_app_u)]
                    selected_applicant_scores = np.append(selected_scores_u, selected_scores_v)
                    kth = int(role_model_ratio*len(selected_applicant_scores))
                    cut_off_role_model_score = np.sort(selected_applicant_scores)[::-1][kth-1]
                    # Fraction of group u among role model candidates for institution k 
                    role_model_u[k] = np.sum(selected_scores_u >=cut_off_role_model_score)/(role_model_ratio*A_k[k])
                    role_model[k][i] = role_model_u[k]
                
                remnant_app_total = remnant_app_total - A_k[k]
                remnant_app_u = remnant_app_u - accepted_app_u
                remnant_app_v = remnant_app_v - (A_k[k] - accepted_app_u)

            # calculates the weighted average of actions
            if reinf_method=='positive':
                opt_act_weighted = np.dot(optimal_actions,c)/np.sum(c)
                update = (opt_act_weighted - N_u/N_t)
            elif reinf_method=='weight':
                opt_act_weighted = np.dot(optimal_actions,weights*c)/np.sum(weights*c)
                update = (opt_act_weighted - N_u/N_t)
            elif (reinf_method=='role_model'):
                opt_act_weighted = np.dot(role_model_u, c)/np.sum(c)
                update = (opt_act_weighted - N_u/N_t)
            elif reinf_method=='indiv_pure_pos_reinf':
                update = np.dot((optimal_actions-N_u/N_t), weights)/np.sum(weights)
            elif reinf_method=='order':
                opt_act_weighted = np.dot(optimal_actions,c)/np.sum(c)
                update = (opt_act_weighted - N_u/N_t)
                update = np.sign(update)*np.power(np.abs(update),order)
            else:
                raise NotImplementedError
            
            # updates the mean parameter
            theta = min(max(epsilon, theta+eta*update),1-epsilon)
            theta_vec[i+1] = theta

        # cumulates the results from the instances
        applicants_u_avg += applicants_u
        admitted_u_avg += admitted_u
        theta_vec_avg += theta_vec
        role_model_avg += role_model

    # calculates the average over the instances
    print('CMFG Setting, all instances are finished.')
    applicants_u_avg = applicants_u_avg/num_instances
    admitted_u_avg = admitted_u_avg/num_instances
    theta_vec_avg = theta_vec_avg/num_instances
    role_model_avg = role_model_avg/num_instances

    return applicants_u_avg, admitted_u_avg, theta_vec_avg, role_model_avg


def plot_results_pos(applicants_u,admitted_u,theta_vec,title,mark=25):
    plt.rc('text', usetex = True)
    plt.rc('font', family='serif')
    width = 6
    height = 4
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    fig.set_size_inches(width, height) #exact size of the figure
    fig.set_dpi(250) 
    lw = 1
    K = applicants_u.shape[0]
    colors = ['#DBB40C','red','blue','tab:green']
    marker = ['o','d','s']
    for i in range(K):
        ax.plot(admitted_u[i], '--', marker=marker[i],markevery=mark, label='Admitted to inst. $k$=' + str(i+1), linewidth=lw, c=colors[i+1],markersize=4.5)
    ax.plot(applicants_u[i], '--', label='Applicants state, $s_t$', linewidth=lw+1, c='black')
    ax.plot(theta_vec, label='Mean parameter ' +r'$\theta_t$' , linewidth=lw+1.5, c=colors[0])
    ax.set_title(title)
    ax.grid()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Proportion')
    ax.legend()
    return fig               

def plot_results_role_model(applicants_u,admitted_u,role_model,theta_vec,title,role_model_p=False):
    plt.rc('text', usetex = True)
    plt.rc('font', family='serif')
    width = 4
    height = 4
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    fig.set_size_inches(width, height) #exact size of the figure
    fig.set_dpi(250) 
    lw = 1
    K = applicants_u.shape[0]
    colors = ['#DBB40C','red','blue','tab:green']
    marker = ['o','d','s']
    for i in range(K):
        if role_model_p is True:
            ax.plot(role_model[i], '--', marker=marker[i],markevery=5, label='Role Model inst. $k$=' + str(i+1), linewidth=lw, c=colors[i+1],markersize=4.5, markerfacecolor='none')
        else:
            ax.plot(admitted_u[i], '--', marker=marker[i],markevery=5, label='Admitted to inst. $k$=' + str(i+1), linewidth=lw, c=colors[i+1],markersize=4.5)

    ax.plot(applicants_u[i], '--', label='Applicants state, $s_t$', linewidth=lw+1, c='black')
    ax.plot(theta_vec, label='Mean parameter ' +r'$\theta_t$' , linewidth=lw+1.5, c=colors[0])
    ax.set_title(title)
    ax.grid()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Proportion')
    ax.legend()
    return fig

def plot_results_diff_lambdas(theta_vec_noncoop, lambdas, title):
    plt.rc('text', usetex = True)
    plt.rc('font', family='serif')
    width = 6
    height = 4
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    fig.set_size_inches(width, height) #exact size of the figure
    fig.set_dpi(250) 
    lw = 2
    colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:red']
    marker = ['o','d','s', 'x']
    for i in range(1, len(lambdas)):
        ax.plot(theta_vec_noncoop[i], color=colors[i-1], label=r'$\lambda=$ '+str(lambdas[i]), linewidth=lw, marker=marker[i-1], markevery=5, markersize=8, markerfacecolor='none')
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Mean parameter '+r'$\theta_t$')
    ax.set_title(title)
    ax.legend()
    ax.grid()
    return fig

def plot_results_percentiles(percentiles_min, percentiles_maj,title):
    plt.rc('text', usetex = True)
    plt.rc('font', family='serif')
    width = 6
    height = 4
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    fig.set_size_inches(width, height) #exact size of the figure
    fig.set_dpi(250) 
    lw = 1
    K = percentiles_min.shape[0]
    colors = ['#DBB40C','red','blue','tab:green']
    marker = ['o','d','s']
    for i in range(K):
        ax.plot(100*percentiles_min[i,:], '--', marker=marker[i],markevery=5, label='Lowest Percentile for Minority inst. $k$=' + str(i+1), linewidth=lw, c=colors[i+1],markersize=4.5, markerfacecolor='none')
        ax.plot(100*percentiles_maj[i,:], marker=marker[i],markevery=5, label='Lowest Percentile for Majority inst. $k$=' + str(i+1), linewidth=lw, c=colors[i+1],markersize=4.5, markerfacecolor='none')
     
    ax.set_title(title)
    ax.grid()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Percentile')
    ax.legend()
    return fig