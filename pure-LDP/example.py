from pure_ldp.frequency_oracles import *
from pure_ldp.heavy_hitters import *

import matplotlib.pyplot as plt

import zipf_dist
import numpy as np
from collections import Counter

import time
import random
import math

import multiprocessing as mp

# Super simple synthetic dataset
#data = np.concatenate(([1] * 8000, [2] * 4000, [3] * 1000, [4] * 500, [5] * 1000, [6] * 1800, [7] * 2000, [8] * 300))
#original_freq = list(Counter(data).values())  # True frequencies of the dataset
num_users = 10000000
#also known as d, domain
num_candidate_strings = 10000
(orig,data) = zipf_dist.zipf(num_users,num_candidate_strings)
for x in range(num_users-len(data)):
    data = np.append(data,random.randint(0,num_candidate_strings-1))
original_freq = list(Counter(data).values())

# Parameters for experiment
epsilon = 4
d = num_candidate_strings
is_the = True
is_oue = True
is_olh = True

def job(f_lock, time_record, final_aggregate, pivot, data, num_candidate_strings, num_index, epsilon):
    client_olh = LHClient(epsilon=epsilon, d=num_candidate_strings,g = 56, use_olh=True)
    server_olh = LHServer(epsilon=epsilon, d=num_candidate_strings,g = 56, use_olh=True)
    search_list = [(i,data[i]) for i in range(len(data)) if i%pivot[1] == pivot[0]]
    client_time = 0
    server_time = 0
    for x in search_list:
        i = x[0]; D = x[1]
        
        ctime = time.time()
        priv_data = client_olh.privatise(D)
        client_time += time.time() - ctime
        
        stime = time.time()
        server_olh.aggregate(priv_data)
        server_time += time.time() - stime

    f_lock.acquire()
    stime = time.time()
    for i in range(0, num_candidate_strings):
        final_aggregate[i]+=server_olh.aggregated_data[i]
    server_time += time.time() - stime

    time_record[0]+=client_time
    time_record[1]+=server_time
    f_lock.release()

'''
# Optimal Local Hashing (OLH)
client_olh = LHClient(epsilon=epsilon, d=d, use_olh=True)
server_olh = LHServer(epsilon=epsilon, d=d, use_olh=True)


# Optimal Unary Encoding (OUE)
client_oue = UEClient(epsilon=epsilon, d=d, use_oue=True)
server_oue = UEServer(epsilon=epsilon, d=d, use_oue=True)

# Threshold Histogram Encoding (THE)
client_the = HEClient(epsilon=epsilon, d=d)
server_the = HEServer(epsilon=epsilon, d=d, use_the=is_the)

# Hadamard Response (HR)
server_hr = HadamardResponseServer(epsilon, d)
client_hr = HadamardResponseClient(epsilon, d, server_hr.get_hash_funcs())'

# RAPPOR
#f=0.24
server_rappor = RAPPORServer(f=0.24, m=248, k=2, num_of_cohorts=8, d=d, lasso=True)
#F = server_rappor.convert_eps_to_f(4)
#print(F)
#server_rappor.update_params(f=F)
server_rappor._generate_hash_funcs()
client_rappor = RAPPORClient(f=0.24, m=248, num_of_cohorts=8, hash_funcs=server_rappor.get_hash_funcs())

# Apple's Count Mean Sketch (CMS)
k = 262144 # 128 hash functions --> changed to 2^18 hash functions
m = 56 # Each hash function maps to the domain {0, ... 1023} --> changed to d=56

server_cms = CMSServer(epsilon, k, m)
client_cms = CMSClient(epsilon, server_cms.get_hash_funcs(), m)
'''
t = time.time()
olh_time = [0]*2

# Simulate client-side privatisation + server-side aggregation
# YW : our parallelized version of OLH
num_processes = 40

f_lock = mp.Lock()
num_index = mp.Value('i',num_candidate_strings)
time_record = mp.Array('f',2)
final_aggregate = mp.Array('f',range(num_candidate_strings))
for element in range(num_candidate_strings):
    final_aggregate[element] = 0
for element in range(2):
    time_record[element] = 0
processes = []
for i in range(num_processes):
    pivot = (i,num_processes)
    p = mp.Process(target = job, args = (f_lock, time_record, final_aggregate, pivot, data, num_candidate_strings, num_index, epsilon, ))
    processes.append(p)
    p.daemon = True
    p.start()

for p in processes:
    p.join()

t = time.time()
# Note instead, we could use server.aggregate_all(list_of_privatised_data) see the Apple CMS example below

# Simulate server-side estimation
#oue_estimates = []
g = int(round(math.exp(epsilon))) + 1
p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)

a = g / ( p * g - 1)
b = num_users / ( p * g - 1)

final_estimation = np.zeros(num_candidate_strings) #[0]*num_candidate_strings;
for idx in range(num_candidate_strings):
    final_estimation[idx] = a*final_aggregate[idx] - b
olh_estimates = final_estimation

print("OLH(server) : ",time_record[0]+time.time()-t)
print("OLH(client-average) : ",time_record[1]/float(num_users))

'''
mse_arr = np.zeros(5)
rappor_client_time = 0
rappor_server_time = 0

for item in data:
    client_time = time.time()
    priv_rappor_data   = client_rappor.privatise(item)
    rappor_client_time += time.time()-client_time
    server_time = time.time()
    server_rappor.aggregate(priv_rappor_data)
    rappor_server_time += time.time()-server_time
    
server_time = time.time()
server_rappor.check_and_update_estimates()
rappor_estimates = server_rappor.estimated_data
rappor_server_time += time.time()-server_time

print("RAPPOR(client-average) : ",rappor_client_time/float(num_users))
print("RAPPOR(server) : ",rappor_server_time)
# Note in the above we could do server.estimate_all(range(1, d+1)) to save looping, see the apple CMS example below

cms_client_time = 0
cms_server_time = 0
'''
# ------------------------------ Apple CMS Example (using aggregate_all and estimate_all) -------------------------
'''
#priv_data = [client_cms.privatise(item) for item in data]
#server_cms.aggregate_all(priv_data)
#cms_estimates = server_cms.estimate_all(range(1, d+1))
for item in data:
    client_time = time.time()
    priv_data = client_cms.privatise(item)
    cms_client_time += time.time()-client_time
    server_time = time.time()
    server_cms.aggregate(priv_data)
    cms_server_time += time.time()-server_time

server_time = time.time()
cms_estimates = server_cms.estimate_all(range(1, d+1))
cms_server_time += time.time()-server_time

print("CMS(client-average) : ",cms_client_time/float(num_users))
print("CMS(server) : ",cms_server_time)
'''
# ------------------------------ Experiment Output (calculating variance) -------------------------


for i in range(0, 1000):
    #mse_arr[0] += (olh_estimates[i] - original_freq[i]) ** 2
    mse_arr[1] += (rappor_estimates[i] - original_freq[i]) ** 2
    #mse_arr[2] += (cms_estimates[i] - original_freq[i]) ** 2
    

print(list(rappor_estimates))
print(original_freq)

mse_arr = mse_arr / d

print("\n")
print("Experiment run on a dataset of size", len(data), "with d=", d, "and epsilon=", epsilon, "\n")
print("Optimised Local Hashing (OLH) Variance: ", mse_arr[0])
print("RAPPOR Variance: ", mse_arr[1])
print("Apple CMS Variance: ",mse_arr[2])
'''
#print("Optimised Unary Encoding (OUE) Variance: ", mse_arr[1])
#print("Threshold Histogram Encoding (THE) Variance: ", mse_arr[2])
#print("Hadamard response (HR) Variance:", mse_arr[3])
#print(sum(hr_estimates))

print("Apple CMS Variance:", mse_arr[4])
print("\n")
#print("Original Frequencies:", original_freq)
#print("OLH Estimates:", olh_estimates)
#print("OUE Estimates:", oue_estimates)
#print("THE Estimates:", the_estimates)
#print("HR Estimates:", hr_estimates)
#print("CMS Estimates:", cms_estimates)
#print("RAPPOR Estimates:",rappor_estimates)
np.save("original",original_freq)
np.save("olh",olh_estimates)
np.save("cms",cms_estimates)
np.save("rappor",rappor_estimates)
print("Note: We round estimates to the nearest integer")

plt.xscale("log")

plt.plot(list(range(num_candidate_strings)), original_freq[:num_candidate_strings],label="original")
plt.plot(list(range(num_candidate_strings)), olh_estimates[:num_candidate_strings],label="OLH",alpha=0.5)
plt.plot(list(range(num_candidate_strings)), rappor_estimates[:num_candidate_strings],label="RAPPOR",alpha=0.5)
plt.plot(list(range(num_candidate_strings)), cms_estimates[:num_candidate_strings],label="CMS",alpha=0.5)
plt.legend()
plt.savefig('result.png')

# ------------------------------ Heavy Hitters - PEM Simulation -------------------------

pem_client = PEMClient(epsilon=3, start_length=2, max_string_length=6, fragment_length=2)
pem_server = PEMServer(epsilon=3, start_length=2, max_string_length=6, fragment_length=2)

s1 = "101101"
s2 = "111111"
s3 = "100000"
s4 = "101100"

print("\nRunning Prefix Extending Method (PEM) to find heavy hitters")
print("Finding top 3 strings, where the alphabet is:", s1, s2, s3, s4)

data = np.concatenate(([s1] * 8000, [s2] * 4000, [s3] * 1000, [s4] * 500))

for index, item in enumerate(data):
    priv = pem_client.privatise(item)
    pem_server.aggregate(priv)

# Can either specify top-k based or threshold based
    # Threshold of 0.05 means we find any possible heavy hitters that have a frequency >= 5%
    # Top-k of three means we try to find the top-3 most frequent strings

heavy_hitters, frequencies = pem_server.find_heavy_hitters(threshold=0.05)
print("Top strings found are:", heavy_hitters, " with frequencies", frequencies)
'''
