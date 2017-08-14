#!/usr/bin/env python

import caffe
import string

import numpy as np
import subprocess

import heapq
import os, sys, stat

# # the last 1000 have the largest loss weight.
# class PriorityQueue:
	# def __init__(self):
		# self._queue = []
		# self._index = 0
	# def push(self, item, priority):
		# heapq.heappush(self._queue, (-priority, self._index, item))
		# self._index += 1
	# def pop(self):
		# return heapq.heappop(self._queue)[-1]
		

# # weight * variation
# PRETRAIN_FILE = 'vgg.caffemodel'
# for s in range(16,17):
	# MODEL_FILE = 'vgg-A.prototxt'
	# np.set_printoptions(threshold='nan')
	# net1 = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
	# top1000 = PriorityQueue()
	# params_txt0 = 'vgg_fault'+str(s)+'/'+'0.txt'
	# params_txt1 = 'vgg_fault'+str(s)+'/'+'1.txt'
	# pf0 = open(params_txt0, 'r')
	# pf1 = open(params_txt1, 'r')

	# # for param_name in net1.params.keys():
		# # weight = net1.params[param_name][0].data
		# # bias = net1.params[param_name][1].data	
		# # if len(weight.shape)==4:
			# # for x in range(weight.shape[0]):
				# # for y in range(weight.shape[1]):
					# # for z in range(weight.shape[2]):
						# # for m in range(weight.shape[3]):
							# # org_weight = weight[x,y,z,m]
							# # variation0=string.atof(pf0.readline())
							# # variation1=string.atof(pf1.readline())
							# # if abs(variation0 - 1) > abs(variation1 - 1):
								# # delta_weight = weight[x,y,z,m] * variation1
							# # else:
								# # delta_weight = weight[x,y,z,m] * variation0
							# # #net1.params[param_name][0].data[x,y,z,m]= delta_weight
							# # loss_weight = abs(org_weight - delta_weight)
							# # top1000.push([param_name,org_weight,variation1,x,y,z,m],loss_weight)
							
		# # elif len(weight.shape)==3:
			# # for x in range(weight.shape[0]):
				# # for y in range(weight.shape[1]):
					# # for z in range(weight.shape[2]):
						# # org_weight = weight[x,y,z]
						# # variation0=string.atof(pf0.readline())
						# # variation1=string.atof(pf1.readline())
						# # if abs(variation0 - 1) > abs(variation1 - 1):
							# # delta_weight = weight[x,y,z] * variation1
						# # else:
							# # delta_weight = weight[x,y,z] * variation0
						# # #net1.params[param_name][0].data[x,y,z,m]= delta_weight
						# # loss_weight = abs(org_weight - delta_weight)
						# # top1000.push([param_name,org_weight,variation1,x,y,z],loss_weight)
						
		# # elif  len(weight.shape)==2:
			# # for x in range(weight.shape[0]):
				# # for y in range(weight.shape[1]):
					# # org_weight = weight[x,y]
					# # variation0=string.atof(pf0.readline())
					# # variation1=string.atof(pf1.readline())
					# # if abs(variation0 - 1) > abs(variation1 - 1):
						# # delta_weight = weight[x,y] * variation1
					# # else:
						# # delta_weight = weight[x,y] * variation0
					# # #net1.params[param_name][0].data[x,y,z,m]= delta_weight
					# # loss_weight = abs(org_weight - delta_weight)
					# # top1000.push([param_name,org_weight,variation1,x,y],loss_weight)
					
		# # for x in range(bias.shape[0]):
			# # org_weight = bias[x]
			# # variation0=string.atof(pf0.readline())
			# # variation1=string.atof(pf1.readline())
			# # if abs(variation0 - 1) > abs(variation1 - 1):
				# # delta_weight = bias[x] * variation1
			# # else:
				# # delta_weight = bias[x] * variation0
			# # #net1.params[param_name][0].data[x,y,z,m]= delta_weight
			# # loss_weight = abs(org_weight - delta_weight)
			# # top1000.push([param_name,org_weight,variation1,x],loss_weight)     
			
	# net1.save('16_test.caffemodel')
	# pf0.close()
	# pf1.close()

	# # keep the large loss weight stable
	# net1 = caffe.Net(MODEL_FILE, '16_test.caffemodel', caffe.TEST)
	
	# for jj in range(1000):
		# original_data = top1000.pop()
		# if len(original_data)==7:
			# net1.params[original_data[0]][0].data[original_data[3],original_data[4],original_data[5],original_data[6]] = 0
		# if len(original_data)==6:
			# net1.params[original_data[0]][0].data[original_data[3],original_data[4],original_data[5]] = 0
		# if len(original_data)==5:
			# net1.params[original_data[0]][0].data[original_data[3],original_data[4]] = 0
		# if len(original_data)==4:
			# net1.params[original_data[0]][1].data[original_data[3]] = 0
	# net1.save('16_test.caffemodel')

	# do the training process
subprocess.call('../build/tools/caffe train -solver vgg_solver_adadelta_01_100000.prototxt -gpu 0'.split())

	
	# os.chmod("vgg_adadelta_iter_30000.caffemodel", stat.S_IRWXU|stat.S_IRGRP|stat.S_IROTH)
	# # transform .h5 to caffemodel
	# np.set_printoptions(threshold='nan')
	# #MODEL_FILE = 'cifar10_quick.prototxt'
	# CHANGED_FILE = 'vgg_adadelta_iter_30000.caffemodel'
	# net1 = caffe.Net(MODEL_FILE, CHANGED_FILE, caffe.TEST)
	# net1.save('16_test.caffemodel')
	
	# PRETRAIN_FILE = '16_test.caffemodel'
	# np.set_printoptions(threshold='nan')
	# net1 = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
	# params_txt0 = 'vgg_fault'+str(s)+'/'+'0.txt'
	# params_txt1 = 'vgg_fault'+str(s)+'/'+'1.txt'
	# pf0 = open(params_txt0, 'r')
	# pf1 = open(params_txt1, 'r')


	# for param_name in net1.params.keys():
		# weight = net1.params[param_name][0].data
		# bias = net1.params[param_name][1].data	
		# if len(weight.shape)==4:
			# for x in range(weight.shape[0]):
				# for y in range(weight.shape[1]):
					# for z in range(weight.shape[2]):
						# for m in range(weight.shape[3]):
							# variation0=string.atof(pf0.readline())
							# variation1=string.atof(pf1.readline())
							# if abs(variation0 - 1) > abs(variation1 - 1):
								# delta_weight = weight[x,y,z,m] * variation1
							# else:
								# delta_weight = weight[x,y,z,m] * variation0
							# net1.params[param_name][0].data[x,y,z,m]= delta_weight
							
		# elif len(weight.shape)==3:
			# for x in range(weight.shape[0]):
				# for y in range(weight.shape[1]):
					# for z in range(weight.shape[2]):
						# variation0=string.atof(pf0.readline())
						# variation1=string.atof(pf1.readline())
						# if abs(variation0 - 1) > abs(variation1 - 1):
							# delta_weight = weight[x,y,z] * variation1
						# else:
							# delta_weight = weight[x,y,z] * variation0
						# net1.params[param_name][0].data[x,y,z]= delta_weight
						
		# elif  len(weight.shape)==2:
			# for x in range(weight.shape[0]):
				# for y in range(weight.shape[1]):
					# variation0=string.atof(pf0.readline())
					# variation1=string.atof(pf1.readline())
					# if abs(variation0 - 1) > abs(variation1 - 1):
						# delta_weight = weight[x,y] * variation1
					# else:
						# delta_weight = weight[x,y] * variation0
					# net1.params[param_name][0].data[x,y]= delta_weight
					
		# for x in range(bias.shape[0]):
			# variation0=string.atof(pf0.readline())
			# variation1=string.atof(pf1.readline())
			# if abs(variation0 - 1) > abs(variation1 - 1):
				# delta_weight = bias[x] * variation1
			# else:
				# delta_weight = bias[x] * variation0
			# net1.params[param_name][1].data[x]= delta_weight
	
	# net1.save('test.caffemodel')
	# pf0.close()
	# pf1.close()
	# subprocess.call('../build/tools/caffe test -model vgg-A.prototxt -weights test.caffemodel -gpu 0'.split())

