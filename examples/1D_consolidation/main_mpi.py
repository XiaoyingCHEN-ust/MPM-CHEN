# encoding: utf-8
# syntax: python2
import numpy as np
import time
import sys
import os
from mpi4py import MPI
sys.path.append(os.environ["PWD"]+'/../../../build-test/')
from lmpm import mpm

#===========================================================#
#                  FUNCTION                                 #
#===========================================================#
def shutdown(status, comm=MPI.COMM_WORLD):
	shutdown =False
	if comm.rank == 0:
		shutdown = status
	shutdown = comm.bcast(shutdown,root=0)
	if shutdown == True:
		MPI.Finalize()
		sys.exit(0)

#===========================================================#
#                  MAIN INPUT                               #
#===========================================================#
input_file = ""  #Column-MPM2d"
dim = 2
pure_mpm = True
checkpoint_factor = int(50000)		    #save all rve with interval factor*output_steps
dem = "yade"                            #"yade"  or "sudo"
json_file = "semi_THM_mpm"                       #e.g. "mpm_stable", "mpm"
unified_rve = True                     #same rve?
rve_id_file = "packing_num.txt"         #if not unified_rve, please specify rve_id_file
target_rve_file = None  	#rve to be saved, if not, use None
target_step = []               #step target rve will be save
rve  = "0.yade.gz"                  #load rve for all material pt.
monitor_interval = 10000               #monitor time interval
monitor_file = "elapse_time.txt"
  

#===========================================================#
#                   MPI CONTROL                             #
#===========================================================#
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

name = MPI.Get_processor_name()    # The name of my node.
hosts = mpi_comm.allgather(name)   # Get the names of all the other hosts
allrank = mpi_comm.allgather(mpi_rank)

#===========================================================#
#                   ANALYSIS PATH                           #
#===========================================================#
path = os.environ["PWD"] #/Explicit-twophase"
msg = ["mpm", "-f", path+input_file+"/", "-i", json_file+".json","-p", "24"]
# rve_save_dir = path+input_file+"/rve_saved/"
#if mpi_rank ==0:
#	if not os.path.exists(rve_save_dir):
#		os.mkdir(rve_save_dir)

#===========================================================#
#                  PURE MPM                                 #
#===========================================================#
if pure_mpm:
	if mpi_rank ==0:
		mpm_solver = mpm()
		mpm_solver.initialize(msg)
		mpm_solver.solve()
		sys.exit()
	else:
		sys.exit()

#===========================================================#
#                   MPM-DEM                                 #
#===========================================================#
if dem == "yade":
	from yade_dem import *
	dem_solver = yade_dem(dim,path+input_file)
elif dem == "sudo":
	from sudo_dem import *
	dem_solver = sudo_dem(dim,path+input_file)
else:
	raise NameError('DEM Solver is not correct')

#===========================================================#
#                   PREPROCESS                              #
#===========================================================#
if mpi_rank==0:
	mpm_solver = mpm()          #create mpm solver object
	mpm_solver.initialize(msg)  #initialize mpm analysis solver,eg. MPM explicit
	info=mpm_solver.get_info()  #info = [dim(unsigned),resume(bool),checkpoint_step(unsigned)]
	mpm_solver.pre_process()    #create mesh, particle, node or resume

	rve_id_global = np.asarray(mpm_solver.send_ids_task(), dtype='int32')
	n_rve_global = len(rve_id_global)
	rve_id_split = np.array_split(rve_id_global, mpi_size)
	print "n_rve_global : ",n_rve_global

	#check unified RVE? if not need packing_num
	if not unified_rve:
		packing_num = np.loadtxt(path+input_file+"/"+rve_id_file,dtype='int32') #for packing identification
	else:
		packing_num = None

	# sendcounts and offset used for Scatterv and Gatherv
	sendcounts = np.asarray([len(i) for i in rve_id_split], dtype='int32')
	offset = np.add.accumulate(sendcounts[:-1])
	offset = np.insert(offset,0,0)

	if (len(target_step) !=0):
		target_rve = np.loadtxt(path+input_file+"/"+target_rve_file,dtype='int32')
	else:
		target_rve = None
else:
	rve_id_global = None
	sendcounts = None
	offset = None
	info = None
	packing_num = None
	target_rve = None

# boardcast sendcounts and offset
sendcounts = mpi_comm.bcast(sendcounts, root=0)    # used for Scatterv and Gatherv 
offset = mpi_comm.bcast(offset, root=0)            # used for Scatterv and Gatherv           
info=mpi_comm.bcast(info, root=0) #info = [dim(unsigned),resume(bool),checkpoint_step(unsigned)]

packing_num = mpi_comm.bcast(packing_num, root=0)
target_rve = mpi_comm.bcast(target_rve, root=0)

resume = info[1]
resume_step=info[2]

rve_id_local = np.zeros(sendcounts[mpi_rank], dtype='int32')
mpi_comm.Scatterv([rve_id_global, tuple(sendcounts), tuple(offset), MPI.INT], rve_id_local)

dem_solver.set_rve_id(rve_id_local)

if mpi_rank==0:
	f = open(path+input_file+"/"+monitor_file, "w")
	f.write("step\t get_defo\t shear_rve\t get_result\t update_state \n")

#===========================================================#
#                  RVE LOADING                              #
#===========================================================#
#mpi_comm.Barrier()
load_rve_begin = MPI.Wtime()

if resume:
	dem_solver.load_rve_resume(resume_step)
else:
	dem_solver.load_rve(unified_rve,rve,packing_num)

#===========================================================#
#                  MPI INFO                                 #
#===========================================================#

if mpi_rank==0:
	print "[MPI INFORMATION] MPI rank: {} \n".format(allrank)
	print "[MPI INFORMATION] MPI host: {} \n".format(hosts)
	print "[MPI INFORMATION] Loading rve cost time {} sec.\n".format(MPI.Wtime()-load_rve_begin)


#===========================================================#
#                  MAIN LOOP                                #
#===========================================================#
while True:
	#=======================================================#
	#     MAINLOOP: STEP, NSTEPS, OUTPUTSTEP                #
	#=======================================================#
	if mpi_rank == 0:
		status = mpm_solver.get_status() #status = [dt, step, nsteps, outputstep] 
	else:
		status = None
	
	status=mpi_comm.bcast(status,root=0)
	step = int(status[1])
	nsteps = int(status[2])
	outputstep = int(status[3])

	#Shut down MPI if finished 
	shutdown(step>=nsteps)

	output = (step%outputstep==0)
	checkpoint = (step % (checkpoint_factor*outputstep) ==0) and (step !=0) #save rve or not
	

	#=======================================================#
	#             MAINLOOP: GET DEFORMATION                 #
	#=======================================================#
	t1_get_deformation = MPI.Wtime()
	#Get displacement gradient and distribute to all ranks
	disp_grad_global = None
	if mpi_rank == 0:
		mpm_solver.get_deformation_task()
		mpm_solver.send_ids_task()
		disp_grad_global = np.asarray(
			mpm_solver.send_deformations_task(), dtype='float64').reshape((-1, dim**2))
	
	disp_grad_local = np.zeros((sendcounts[mpi_rank], dim**2), dtype='float64')
	mpi_comm.Scatterv([disp_grad_global, tuple(sendcounts*dim**2),
					tuple(offset*dim**2), MPI.DOUBLE], disp_grad_local, root=0)


	#=======================================================#
	#         MAINLOOP: SHEAR RVE,  (SAVE) RVE              #
	#=======================================================#
	t2_shear_rve = MPI.Wtime()
	#Shear rve in all rank
	result = []
	for i in range(len(rve_id_local)):
		temp = dem_solver.shear_rve(i, disp_grad_local[i], output)
		result.append(temp)  #result=[s,n,r,fabric_CN,fabric_PO]

	# if need, save all rve for further resume
	if (checkpoint and (step!=resume_step)) :
		if mpi_rank == 0:
			if not os.path.exists(rve_save_dir+"step_"+str(step)):
				os.mkdir(rve_save_dir+"step_"+str(step))
		mpi_comm.Barrier()
		dem_solver.save_all_rve(step)

	# if need, save target rve for local analysis
	if (step in target_step):
		if mpi_rank == 0:
			if not os.path.exists(rve_save_dir+"step_"+str(step)):
				os.mkdir(rve_save_dir+"step_"+str(step))
		mpi_comm.Barrier()
		dem_solver.save_rve(target_rve, step)


	#=======================================================#
	#     MAINLOOP: GET STRESS, POROSITY,ROTATION,FABRIC    #
	#=======================================================#
	mpi_comm.Barrier()
	t3_get_result = MPI.Wtime()
	# retrive stress 
	stress = [i[0] for i in result]
	stress = np.asarray(stress, dtype='float64')
	stress_gather = None
	if mpi_rank == 0:
		stress_gather = np.zeros((n_rve_global, 6), dtype='float64')
	
	mpi_comm.Gatherv(stress, [stress_gather, tuple(
		sendcounts*6), tuple(offset*6), MPI.DOUBLE], root=0)

	#output porosity, rotation, fabric, if needed
	if output:
		porosity = [i[1] for i in result]
		rotation = [i[2] for i in result]
		fabric_CN = [i[3] for i in result]

		porosity = np.asarray(porosity, dtype='float64')
		rotation = np.asarray(rotation, dtype='float64')
		fabric_CN = np.asarray(fabric_CN, dtype='float64')

		porosity_gather = None
		rotation_gather = None
		# fabric_CN_gather = None

		if mpi_rank == 0:
			porosity_gather = np.zeros((n_rve_global, 1), dtype='float64')
			rotation_gather = np.zeros((n_rve_global, 3), dtype='float64')
			# fabric_CN_gather = np.zeros((n_rve_global, dim**2), dtype='float64')

		mpi_comm.Gatherv(porosity, [porosity_gather, tuple(
			sendcounts*1), tuple(offset*1), MPI.DOUBLE], root=0)
		mpi_comm.Gatherv(rotation, [rotation_gather, tuple(
			sendcounts*3), tuple(offset*3), MPI.DOUBLE], root=0)
		# mpi_comm.Gatherv(fabric_CN, [fabric_CN_gather, tuple(
		# 	sendcounts*dim**2), tuple(offset*dim**2), MPI.DOUBLE], root=0)

	#=======================================================#
	#     MAINLOOP: SEND RESULT, UPDATE MPM                 #
	#=======================================================#

	t4_update_state = MPI.Wtime()
	##Update particle state
	shutdown_status = False
	if mpi_rank == 0:
		stress_gather = stress_gather.flatten().tolist()
		
		# Bool stands for incremental
		mpm_solver.set_stress_task(stress_gather, False)
		if output:
			mpm_solver.set_porosity_task(
				porosity_gather.flatten().tolist())
			mpm_solver.set_rotation_task(
				rotation_gather.flatten().tolist())
			# mpm_solver.set_fabric_CN_task(
			#     fabric_CN_gather.flatten().tolist())

		#update_status, True: update success, False:Particle outside mesh 
		update_status = mpm_solver.update_state_task()
		shutdown_status= not update_status
	shutdown(shutdown_status)

	#=======================================================#
	#     MAINLOOP: SAVE MONITOR DATA                       #
	#=======================================================#
	t5_end = MPI.Wtime()
	if mpi_rank == 0:
		if (step % monitor_interval ==0):
			t1=t2_shear_rve - t1_get_deformation    #get_deformation
			t2=t3_get_result - t2_shear_rve         #shear_rve
			t3=t4_update_state -t3_get_result       #get_result_send_result
			t4=t5_end - t4_update_state             #update_result
			content = "{}\t{:7f}\t{:7f}\t{:7f}\t{:7f}\n".format(step,t1,t2,t3,t4)
			f.write(content)
