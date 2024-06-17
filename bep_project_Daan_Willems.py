import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag, coo_matrix
import pymatching
import sinter
import stim
import time
from scipy.optimize import curve_fit, fsolve, root_scalar
import itertools
import pymatching._cpp_pymatching as _cpp_pm
from pymatching import color_code_stim
from typing import Union, Callable
from ldpc.codes import rep_code
from ldpc import bp_decoder, bposd_decoder
from bposd.css import css_code
import inspect

def pcm_surface_code(distance: int):
    '''This function returns the parity check matrix of the surface code.
    The parity check matrix is returned as a css_code. The input variable distance
    determines the distance of the parity check matrix.'''
    
    def X_to_Z(pcm, distance):
        for row in range(len(pcm)):
            matrix = np.reshape(pcm[row], (distance, distance))
            transposed_matrix = np.transpose(matrix)
            rotated_matrix = np.flip(transposed_matrix, axis=1)
            rotated_array = rotated_matrix.flatten()   
            pcm[row]=rotated_array
        return pcm
    
    H = np.zeros(distance**2)
    
    for i in range(0,distance-2,2):
        new_row = np.zeros(distance**2)
        new_row[i]=1 
        new_row[i+1]=1
        H = np.vstack((H,new_row))

    for i in range(distance+1,distance**2,2):
        if (i+1)%distance!=0:
            new_row = np.zeros(distance**2)
            new_row[i]=1 
            new_row[i+1]=1
            new_row[i-distance]=1
            new_row[i-distance+1]=1
            H = np.vstack((H,new_row))
                
    for i in range(distance**2-distance+1,distance**2,2):
        new_row = np.zeros(distance**2)
        new_row[i]=1 
        new_row[i+1]=1
        H = np.vstack((H,new_row))
    
    H = np.delete(H, 0, 0)
    return css_code(hx=np.array(H), hz=np.array(X_to_Z(H, distance)))


def pcm_toric_code(distance):
    '''This function returns the parity check matrix of the toric code.
    The parity check matrix is returned as a css_code. The input variable distance
    determines the distance of the parity check matrix.'''
    
    def pcm_repetition_code(distance):
        row_ind, col_ind = zip(*((i, j) for i in range(distance) for j in (i, (i+1)%distance)))
        data = np.ones(2*distance, dtype=np.uint8)
        return csc_matrix((data, (row_ind, col_ind)))

    Hr = pcm_repetition_code(distance)
    Hx = hstack([kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)], dtype=np.uint8)
    Hz = hstack([kron(eye(Hr.shape[1]), Hr), kron(Hr.T, eye(Hr.shape[0]))], dtype=np.uint8)
    Hx.data = Hx.data % 2
    Hz.data = Hz.data % 2
    Hx.eliminate_zeros()
    Hz.eliminate_zeros()
    return css_code(hx=np.array(Hx.toarray()), hz=np.array(Hz.toarray()))


def pcm_bivariate_bicycle(distance):
    '''This function returns the parity check matrix of the bivariate bicycle code.
    The parity check matrix is returned as a css_code. The input variable distance
    determines which combination of l,m,A,B is chosen in order to create the parity
    check matrix. The input variable distance doesn't equal the actual distance of the
    stabilizer code. The name 'distance' was only chosen in order to make it compatible
    with other stablizer code functions.'''
    
    def cyclic_shift_matrix(l):
        matrix = np.zeros((l,l))
        for r in range(l):
            matrix[r][(r+1)%l]=1
        return matrix
    
    def values(l,m):
        if (l==6 and m==6) or (l==9 and m==6) or (l==12 and m==6):
            return 3,1,2,3,1,2
        elif l==15 and m==3:
            return 9,1,2,0,2,7
        elif l==12 and m==12:
            return 3,2,7,3,1,2
        elif l==30 and m==6:
            return 9,1,2,3,25,26
        elif l==21 and m==18:
            return 3,10,17,5,3,19 
        
    lm_list = [[6,6],[15,3],[9,6],[12,6],[12,12],[30,6],[21,18]]
    l, m = lm_list[(distance-3)//2]
    a,b,c,d,e,f = values(l,m)
    Sl = cyclic_shift_matrix(l)
    Sm = cyclic_shift_matrix(m)
    Il = eye(l).toarray()
    Im = eye(m).toarray()
    x = kron(Sl, Im).toarray()
    y = kron(Il, Sm).toarray()
    A = np.linalg.matrix_power(x, a) + np.linalg.matrix_power(y, b) + np.linalg.matrix_power(y, c)
    B = np.linalg.matrix_power(y, d) + np.linalg.matrix_power(x, e) + np.linalg.matrix_power(x, f)

    Hx = hstack([coo_matrix(A), coo_matrix(B)], dtype=np.uint8)
    Hz = hstack([coo_matrix(B.T), coo_matrix(A.T)], dtype=np.uint8)
    Hx.data = (Hx.data % 2)
    Hz.data = (Hz.data % 2)
    H = css_code(hx=np.array(Hx.toarray()),hz=np.array(Hz.toarray()))
    return H  

def pcm_color_code_666(distance):
    '''This function returns the parity check matrix of the 666-color-code.
    The parity check matrix is returned as a css_code. The input variable distance
    determines the distance of the parity check matrix.'''
    
    qubits_per_row = []
    tot_qubits = int((3/4)*distance**2+(1/4))
    
    for i in range(1,distance+1):
        qubits_per_row.append(i)
        if i%2!=0 and i!=distance:
            qubits_per_row.append(i)
    
    tot_qubits_above = np.cumsum(qubits_per_row[:(len(qubits_per_row)-1)])
    tot_qubits_above = np.insert(tot_qubits_above,0,0)

    H = np.zeros(tot_qubits)
    
    for i in range(len(qubits_per_row)-1):
        for j in range(qubits_per_row[i]):
            if i%3==0 and (j%2==1 or j==0):
                new_row = np.zeros(tot_qubits)
                new_row[tot_qubits_above[i]+j]=1
                new_row[tot_qubits_above[i+1]+j]=1 
                new_row[tot_qubits_above[i+2]+j+1]=1    
                a=0
                if j%2==1:
                    a=2
                    new_row[tot_qubits_above[i]+j+1]=1
                    new_row[tot_qubits_above[i+1]+j+1]=1     
                new_row[tot_qubits_above[i+2]+j+a]=1    
                H = np.vstack((H,new_row))  
    
            elif (i-1)%3==0 and j%2==0:
                new_row = np.zeros(tot_qubits)
                new_row[tot_qubits_above[i]+j]=1
                new_row[tot_qubits_above[i+1]+j+1]=1  
                new_row[tot_qubits_above[i+2]+j+1]=1
                new_row[tot_qubits_above[i+2]+j+2]=1
                
                if j!=(qubits_per_row[i]-1):
                    new_row[tot_qubits_above[i]+j+1]=1
                    new_row[tot_qubits_above[i+1]+j+2]=1
                H = np.vstack((H,new_row)) 
              
            elif j%2==0:
                new_row = np.zeros(tot_qubits)
                new_row[tot_qubits_above[i]+j]=1
                new_row[tot_qubits_above[i]+j+1]=1
                new_row[tot_qubits_above[i+1]+j]=1
                new_row[tot_qubits_above[i+1]+j+1]=1
                if i!=(len(qubits_per_row)-2):
                    new_row[tot_qubits_above[i+2]+j]=1
                    new_row[tot_qubits_above[i+2]+j+1]=1
                H = np.vstack((H,new_row)) 
            
    H = np.delete(H, 0, 0)
    return css_code(hx=H, hz=H)

              
def pcm_color_code_488(distance):
    '''This function returns the parity check matrix of the 488-color-code.
    The parity check matrix is returned as a css_code. The input variable distance
    determines the distance of the parity check matrix.'''
    
    qubits_per_row = []
    tot_qubits = int((1/2)*distance**2+distance-(1/2))
    
    for i in range(2,distance,2):
        qubits_per_row.append(i)
        qubits_per_row.append(i)
    qubits_per_row.append(distance)
    
    tot_qubits_above = np.cumsum(qubits_per_row[:(len(qubits_per_row)-1)])
    tot_qubits_above = np.insert(tot_qubits_above,0,0)
    H = np.zeros(tot_qubits)
    
    for i in range(0,distance-2,2):
        for j in range(0,qubits_per_row[i],2):
            if i%4==0:
                new_row_green = np.zeros(tot_qubits)
                new_row_green[tot_qubits_above[i]+j]=1
                new_row_green[tot_qubits_above[i+1]+j]=1
                new_row_green[tot_qubits_above[i+2]+j]=1
                new_row_green[tot_qubits_above[i+2]+j+1]=1
                
                if j!=0:
                    new_row_green[tot_qubits_above[i]+j-1]=1
                    new_row_green[tot_qubits_above[i+1]+j-1]=1
                    new_row_green[tot_qubits_above[i-1]+j-1]=1
                    new_row_green[tot_qubits_above[i-1]+j-2]=1
                H = np.vstack((H,new_row_green))

            new_row_red = np.zeros(tot_qubits)
            new_row_red[tot_qubits_above[i]+j]=1
            new_row_red[tot_qubits_above[i]+j+1]=1
            new_row_red[tot_qubits_above[i+1]+j]=1
            new_row_red[tot_qubits_above[i+1]+j+1]=1
            H = np.vstack((H,new_row_red))
                           
            if i%4==2:
                if i==(distance-3):
                    a=1
                else:
                    a=0
                    
                new_row_blue = np.zeros(tot_qubits)
                new_row_blue[tot_qubits_above[i]+j+1]=1
                new_row_blue[tot_qubits_above[i+1]+j+1]=1
                new_row_blue[tot_qubits_above[i+2]+j+2-a]=1
                new_row_blue[tot_qubits_above[i+2]+j+3-a]=1
                
                if j!=(qubits_per_row[i]-2):
                    new_row_blue[tot_qubits_above[i]+j+2]=1
                    new_row_blue[tot_qubits_above[i+1]+j+2]=1
                    new_row_blue[tot_qubits_above[i-1]+j]=1
                    new_row_blue[tot_qubits_above[i-1]+j+1]=1
                H = np.vstack((H,new_row_blue))

    if (distance-1)%4==0:
        a=0
    else:
        a=1
        
    for j in range(0,distance-1,2):
        new_row = np.zeros(tot_qubits)
        new_row[tot_qubits_above[-2]+j]=1
        new_row[tot_qubits_above[-2]+j+1]=1
        new_row[tot_qubits_above[-1]+j+a]=1
        new_row[tot_qubits_above[-1]+j+a+1]=1
        H = np.vstack((H,new_row))

    H = np.delete(H, 0, 0)
    return css_code(hx=H, hz=H)


def count_logical_errors(code, num_shots, p: float = 0.1, q: float = 0.1, distance: int = 3, 
                         degree_of_noise: int = 0, repetitions: int = 10, decoder: str = 'mwpm', X_or_Z: str = 'X'):
    '''This function counts the amount of logical errors made over a number of simulations (=num_shots)
    for a particular stabilizer code (=code). The function works for two different decoders, BP and MWPM,
    and for 3 different noise models: independent noise, syndrome noise, circuit-level noise 
    (degree_of_noise = 0, 1, 2). All other functions in this function represent a decoder for a specific
    decoder and degree of noise. Lets say decoder = MPWM and degree of noise = 0, the corresponding
    function for this configuration would be mwpm_0noise.'''
    
    assert decoder in ['mwpm', 'bp'], 'decoder should be either "mwpm" or "bp".'
    assert degree_of_noise in [0,1,2], 'degree_of_noise should be either 0, 1 or 2.'
    assert X_or_Z in ['X', 'Z'], '"X_or_Z" should be either "X" or "Z".'
    
    def xz(code, distance, X_or_Z):
        if X_or_Z == 'X':
            return code(distance).hx, code(distance).lx
        elif X_or_Z == 'Z':
            return code(distance).hz, code(distance).lz
        
    def mwpm_0noise(code, distance, p, num_shots, X_or_Z):
        if code==pcm_color_code_666:
            colorcode = color_code_stim.ColorCode(d=distance, rounds=1, p_bitflip=p)
            num_errors = colorcode.simulate(num_shots)
        else:
            H, logical = xz(code, distance, X_or_Z)
            matching = pymatching.Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logical)
            noise = (np.random.random((num_shots, H.shape[1])) < p).astype(np.uint8)
            syndromes = (noise @ H.T) % 2
            actual_observables = ((noise @ logical.T) % 2)
            predicted_observables = matching.decode_batch(syndromes)
            num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
            #num_errors = np.sum(np.abs(actual_observables[:,0]-predicted_observables[:,0]))
        return num_errors
    
    def mwpm_1noise(code, distance, p, q, num_shots, repetitions, X_or_Z):
        if code==pcm_color_code_666:
            colorcode = color_code_stim.ColorCode(d=distance, rounds=1, p_bitflip=p, p_meas=q)
            num_errors = colorcode.simulate(num_shots)
        else:
            H, logical = xz(code, distance, X_or_Z)
            #
            matching = pymatching.Matching(H, weights=np.log((1-p)/p), timelike_weights=np.log((1-q)/q), repetitions=repetitions, faults_matrix=logical)
            num_stabilisers, num_qubits = H.shape
            num_errors = 0
                
            for i in range(num_shots):
                noise_new = (np.random.rand(num_qubits, repetitions) < p).astype(np.uint8)
                noise_cumulative = (np.cumsum(noise_new, 1) % 2).astype(np.uint8)
                noise_total = noise_cumulative[:,-1]
                syndrome = H@noise_cumulative % 2
                syndrome_error = (np.random.rand(num_stabilisers, repetitions) < q).astype(np.uint8)
                syndrome_error[:,-1] = 0 
                noisy_syndrome = (syndrome + syndrome_error) % 2
                noisy_syndrome[:,1:] = (noisy_syndrome[:,1:] - noisy_syndrome[:,0:-1]) % 2
                predicted_logicals_flipped = matching.decode(noisy_syndrome)
                actual_logicals_flipped = noise_total@logical.T % 2
                if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
                    num_errors += 1
        return num_errors
    
    def mwpm_2noise(code, distance, p, repetitions, num_shots, X_or_Z):
        assert code in [pcm_surface_code, pcm_color_code_666], '''Circuit level noise is not supported for this type 
        of code and decoder. Choose either surface code or 6,6,6-color code or a different decoder.'''
        
        if code==pcm_color_code_666:
            colorcode = color_code_stim.ColorCode(d=distance, rounds=1, p_circuit=p)
            num_errors = colorcode.simulate(num_shots)
        else:
            if code==pcm_surface_code and X_or_Z == 'X':
                a = "surface_code:rotated_memory_x"
            elif code==pcm_surface_code and X_or_Z == 'Z':
                a = "surface_code:rotated_memory_z"
            elif code==pcm_color_code_666:
                a = "color_code:memory_xyz"
                
            def create_standard_code(rounds, distance, error_rate, type_of_code: str):        
                circuit = stim.Circuit.generated(
                    type_of_code,
                    rounds=rounds,
                    distance=distance,
                    after_clifford_depolarization=error_rate,
                    after_reset_flip_probability=error_rate,
                    before_measure_flip_probability=error_rate,
                    before_round_data_depolarization=error_rate)
                return circuit
            
            circuit = create_standard_code(repetitions, distance, p, a)
            sampler = circuit.compile_detector_sampler()
            detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
            detector_error_model = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
            matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
            predictions = matcher.decode_batch(detection_events)
            num_errors = 0
            
            for shot in range(num_shots):
                actual_for_shot = observable_flips[shot]
                predicted_for_shot = predictions[shot]
                if not np.array_equal(actual_for_shot, predicted_for_shot):
                    num_errors += 1
        return num_errors            
        
    def bp_0noise(code, distance, num_shots, p, repetitions, X_or_Z):
        H, logical = xz(code, distance, X_or_Z)
        bpd=bposd_decoder(H, error_rate=p, bp_method="ps",osd_order=4)
        noise = (np.random.random((num_shots, H.shape[1])) < p).astype(np.uint8)
        syndromes = (noise @ H.T) % 2
        predicted_observables = []
        for syndrome in syndromes:
            predicted_observables.append(bpd.decode(syndrome))
        residual_error=(predicted_observables+noise) % 2
        #num_errors = int(np.sum(logical@residual_error.T % 2))/logical.shape[0]
        #num_errors = int(np.sum((logical@residual_error.T % 2)[0,:]))
        num_errors = np.sum(np.any(logical@residual_error.T % 2,axis=0))

        return num_errors
    
    def bp_1noise(code, distance, repetitions, p, q, num_shots):
        raise Exception('Belief propagation does not support more complex noise models (for now). Choose degree_of_noise=0.')
    
    def bp_2noise(code, distance, repetitions, p, num_shots):
        raise Exception('Belief propagation does not support more complex noise models (for now). Choose degree_of_noise=0.')
    
    if decoder=='mwpm':
        if degree_of_noise==0:
            num_errors = mwpm_0noise(code, distance, p, num_shots, X_or_Z)   
        elif degree_of_noise==1:
            num_errors = mwpm_1noise(code, distance, p, q, num_shots, repetitions, X_or_Z)  
        elif degree_of_noise==2:
            num_errors = mwpm_2noise(code, distance, p, repetitions, num_shots, X_or_Z)      
    elif decoder=='bp':
        if degree_of_noise==0:
            num_errors = bp_0noise(code, distance, num_shots, p, repetitions, X_or_Z)
        elif degree_of_noise==1:
            num_errors = bp_1noise(code, distance, repetitions, p, q, num_shots)   
        elif degree_of_noise==2:
            num_errors = bp_2noise(code, distance, repetitions, p, num_shots)
    return num_errors

def logical_vs_physical(code: Callable, distances: list = [5,7,9], num_shots: int = 10000, num_data_points: int = 50, 
                        degree_of_noise: int = 0, repetitions: int = 10, 
                        get_time: bool = True, get_threshold: bool = True, get_plot: bool = True, save_fig: bool = False,
                        ps_range: list = [0.001, 0.2], q: float = 0.0,
                        decoder: str = 'mwpm', X_or_Z: str = 'X', custom_fit_func: Union[Callable,bool]=False):
    '''This function calculates the threshold value for a particular stabilizer code, noise model and decoder.
    It calculates the logical error rate with the count_logical_error() function. It then makes a fit of these results
    with either a linear or non-linear function. The intersection of these fits equals the threshold.
    The function has an option to return the threshold value, runtime and a plot of the data including fits.
    When using this function it might be smart to set get_plot=True so you can check whether or not you are looking in 
    the right physical error rate range (ps_range).'''
    
    
    ps = np.linspace(ps_range[0],ps_range[1],num_data_points)
    all_y_axis=[]
    start_time = time.time()

    for i in distances:
        y_axis=[]
        for j in ps: 
            logical_errors = count_logical_errors(code, num_shots, p=j, q=j, distance=i, 
                                                  degree_of_noise=degree_of_noise, repetitions=repetitions, 
                                                  decoder=decoder, X_or_Z = X_or_Z)
            y_axis.append(logical_errors/num_shots)
            
        all_y_axis.append(y_axis)
        if get_plot and get_threshold: 
            plt.scatter(ps, y_axis, label="d=" + str(i), s=20)
        elif get_plot:
            plt.plot(ps, y_axis, label="d=" + str(i))
    
    if custom_fit_func==False:
        def fit_function(x,a,b,c,d):
            return a*np.arctan(b*(x-c))+d
    else:
        fit_function=custom_fit_func
    num_params = len(inspect.signature(custom_fit_func).parameters)-1

    def fit_function_lin(x,a,b):
        return a*x+b
    
    def index_lowest(func):
        lowest_value = 10**10
        lowest_index = 0
        for i in range(len(func)):
            if abs(func[i])<lowest_value:
                lowest_value = func[i]
                lowest_index = i
        return lowest_index
             
    if get_threshold:
        values = []
        for y_axis in all_y_axis:
            try:
                popt = curve_fit(fit_function, ps, y_axis)[0]
                if get_plot:
                    plt.plot(ps, fit_function(ps, *popt))
                    ymax = max(fit_function(ps, *popt))
            except:
                popt = curve_fit(fit_function_lin, ps, y_axis)[0]
                if get_plot:
                    plt.plot(ps, fit_function_lin(ps, *popt))
                    ymax = max(fit_function_lin(ps, *popt))

            values.append(popt)

        param_count=0
        for fit in values:
            if len(fit)==num_params:
                param_count+=1
        if param_count>=2:
            for i in range(len(values)):
                if len(values[i])==2:
                    values.pop(i)        
        else:
            for i in range(len(values)):
                if len(values[i])==num_params:
                    values.pop(i)
  
        combinations = list(itertools.combinations(range(len(values)), 2))
        more_ps = np.linspace(ps_range[0],ps_range[1],50000)
        intercepts = []

        if param_count>=2:
            for i in combinations:
                h = fit_function(more_ps, *values[i[0]])-fit_function(more_ps, *values[i[1]])  
                intercepts.append(more_ps[index_lowest(h)])
        else:
            for i in combinations:
                h = fit_function_lin(more_ps, *values[i[0]])-fit_function_lin(more_ps, *values[i[1]])
                intercepts.append(more_ps[index_lowest(h)])
        threshold = np.median(intercepts)
        
        if get_plot:
            plt.vlines(threshold, 0, ymax, colors='k', linestyles='dashed', label='threshold')
         
            
    if get_plot:
        plt.xlabel("physical error rate")
        plt.ylabel("logical error rate")
        plt.legend()
        if save_fig:
            plt.savefig('bepimage.png', dpi=500, format='png')
        plt.show()    
        
                    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if get_time and get_threshold:
        print('runtime was ' + str(round(elapsed_time,4)) + ' seconds')
        return threshold, elapsed_time
    elif get_time:
        print('runtime was ' + str(round(elapsed_time,4)) + ' seconds')
        return elapsed_time
    elif get_threshold:
        return threshold

def distance_vs_runtime(All: bool=True, Avg: bool=False):
    '''This function plots the runtime of a particular combination of decoder and stabilizer code
    against the distance. If All=True, the function will return a plot showing all runtime vs
    distance curves for every combination of decoder and stabilizer code. If Avg=True,
    the function will return a runtime vs distance plot only for the average runtime per decoder.
    In this case 1 curve for the MWPM algorithm and 1 curve for the BP algorithm.'''
    
    def plot_function(x,a,b,c,d):
        return a*(x**b)
    
    list_decoders = ['mwpm', 'bp']
    
    if All:
        list_codes = [pcm_surface_code,pcm_toric_code,pcm_color_code_488,pcm_color_code_666,pcm_bivariate_bicycle]
        
        for decoder in list_decoders:
            for code in list_codes:
                list_distances = [3,5,7,9,11,13,15,17]
                list_distances_plot = list_distances
                
                if code==pcm_bivariate_bicycle:
                    list_distances = [3,5,7,9,11]
                    list_distances_plot = [6,10,10,12,18]
                    
                more_distances = np.linspace(3, max(list_distances_plot), 1000)
                
                try:
                    list_times = []
                    for distance in list_distances:
                        a = logical_vs_physical(code, num_shots = 200, num_data_points = 5, 
                                    degree_of_noise = 0, distances = [distance], ps_range = [0.02,0.2], decoder=decoder, repetitions=1,
                                    get_threshold = False, get_plot=False, get_time=True) 
                        list_times.append(a)
                        print(distance)
                    
                    #plt.plot(list_distances_plot, list_times, marker = 'o', label=str(code.__name__)[4:] + ', ' + str(decoder))
                    plt.scatter(list_distances_plot, list_times, label=str(code.__name__)[4:] + ', ' + str(decoder))
                    popt = curve_fit(plot_function, list_distances_plot, list_times)[0]
                    plt.plot(more_distances, plot_function(more_distances, *popt))  
                    print(str(code.__name__)+' is done')   
                except:
                    print(str(code.__name__) + ' ' + str(decoder) + ' failed') 
                    
        plt.ylim(bottom=10**(-1.8))
        plt.xlim(left=5)
        plt.xlabel("distance (-)")
        plt.ylabel("runtime (s)")
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.5, 0.8))
        plt.show() 
        plt.savefig('bepimage4all.png', dpi=500, format='png', bbox_inches = 'tight', pad_inches = 3)
    
    elif Avg:
        list_codes = [pcm_surface_code,pcm_toric_code,pcm_color_code_666]
        list_distances = [3,5,7,9,11,13,15,17]
        more_distances = np.linspace(3, max(list_distances), 1000)
        for decoder in list_decoders:
            list_times = []
            for distance in list_distances:
                try:
                    avg_times = []
                    for code in list_codes:
                        a = logical_vs_physical(code, num_shots = 200, num_data_points = 5, 
                                    degree_of_noise = 0, distances = [distance], ps_range = [0.02,0.2], decoder=decoder, repetitions=1,
                                    get_threshold = False, get_plot=False, get_time=True) 
                        avg_times.append(a)
                        print(distance)
                        
                    list_times.append(sum(avg_times)/len(avg_times))  
                    
                      
                except:
                    print(str(code.__name__) + ' ' + str(decoder) + ' failed') 
                    
            plt.scatter(list_distances, list_times, label=str(decoder))
            popt = curve_fit(plot_function, list_distances, list_times)[0]
            plt.plot(more_distances, plot_function(more_distances, *popt))  
            print(str(decoder)+' is done') 
            
        plt.legend()  
        plt.xlabel("distance (-)")
        plt.ylabel("runtime (s)")
        plt.yscale('log')
        plt.savefig('bepimage4avg.png', dpi=500, format='png')
        plt.show()     


def all_thresholds(decoder: str = 'mwpm'):
    '''This function is able to calculate the threshold value for all stabilizer codes
    for a particular decoder.'''
    list_codess = [pcm_surface_code,pcm_toric_code,pcm_color_code_488,pcm_color_code_666,pcm_bivariate_bicycle]
    threshold_dict = {'decoder':decoder}
    for code in list_codess:
        try:
            threshold, time = logical_vs_physical(code, num_shots = 20000, num_data_points = 40, repetitions=1, 
                               degree_of_noise = 0, distances = [5,7,9], ps_range = [0.01,0.2], decoder=decoder)
            threshold_dict.update({code.__name__:threshold})
            print(str(code.__name__)+' succeeded')
        except:
            print(str(code.__name__)+' failed')
        
    return threshold_dict


def threshold_vs_repetitions(q_list):
    '''This function plots the threshold value against the amount of repetitions
    for multiple syndrome error rates q. The input variable q_list determines
    over which values of q the function will iterate. Before you run this function,
    it is important to check in the function logical_vs_physical, where the 
    count_logical_error function is executed, that the input variable q of this function
    is set to q=q instead of q=i.'''
           
    for q in q_list:
        reps = []
        thresholds = []
        for rep in range(1,7):
            threshold = logical_vs_physical(pcm_surface_code, q = q, num_shots = 20000, num_data_points = 30, repetitions=rep, 
                                        degree_of_noise = 1, distances = [3,5,7], ps_range = [0.01,0.12], decoder='mwpm',
                                        get_time=True, save_fig=False, get_plot=False)[0]
            reps.append(rep)
            thresholds.append(threshold)
        plt.plot(reps, thresholds, marker = 'o', label='q = '+str(round(q,3)))
        print('q = '+str(q)+' completed.')
        
    plt.legend()  
    plt.xlabel("repetitions")
    plt.ylabel("threshold value")
    plt.savefig('bepimagethreshold2.png', dpi=500, format='png')
    plt.show()    

def different_noise():
    '''This function calculates the threshold value for different noise levels.
    The function returns a dictionary where every key is a noise level, and every
    value is the threshold corresponding to that noise level. Before you run this function,
    it is important to check in the function logical_vs_physical, where the 
    count_logical_error function is executed, that the input variable q of the 
    count_logical_error function is set to q=i instead of q=q. This ensures that all error 
    rates remain the same.'''
    
    noise_dict = {}
    ps_range_list = [[0.01,0.2],[0.01,0.1],[0.002,0.02]]
    for noise in range(3):
        threshold = logical_vs_physical(pcm_surface_code, num_shots = 10000, num_data_points = 30, repetitions=10, 
                                degree_of_noise = noise, distances = [3,5,7], ps_range = ps_range_list[noise], decoder='mwpm',
                                get_time=True, save_fig=False, get_plot=True, get_threshold=True)[0]
        noise_dict.update({'Noise level '+str(noise):threshold})
        print('Noise level '+str(noise)+' is done.')
        
    return noise_dict


def fit_functiondf(x,a,b,c,d,e):
    return a/(1+np.exp(-b*(x-c)))+d+e

print(logical_vs_physical(pcm_surface_code, num_shots = 10000, num_data_points = 30, 
                            degree_of_noise = 0, distances = [9,11,13], ps_range = [0.05,0.2], decoder='mwpm',
                            custom_fit_func=fit_functiondf)[0])

