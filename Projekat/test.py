#!/usr/bin/python
import sys

DIR = 'C:\\Users\\Dule\\Desktop\\Projekat'
res = []
n = 0
with open(DIR+'\\res1.txt') as file:	
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        if(id>0):
            cols = line.split('\t')
            if(cols[0] == ''):
                continue
            cols[1] = cols[1].replace('\r', '')
            res.append(float(cols[1]))
            n += 1

correct = 0
student1 = []
student2 = []
student_results = []
with open(DIR+"\\out.txt") as file:
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        cols = line.split('\t')
        if(cols[0] == ''):
            continue
        if(id==0):
            student1 = cols  
        if(id==1):
            student2 = cols  
        elif(id>2):
            cols[1] = cols[1].replace('\r', '')
            student_results.append(float(cols[1]))

diff = 0
for index, res_col in enumerate(res):
    diff += abs(res_col - student_results[index])
percentage = 100 - diff/sum(res)*100

print student1
print student2
print 'Procenat tacnosti:\t'+str(percentage)
print 'Ukupno:\t'+str(n)
