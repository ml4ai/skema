f = open("mml.txt").readlines()

d = dict()

for l in f:
  tokens = l.split()
  for t in tokens:
    if t not in d.keys():
      d[t]=1
    else:
      d[t]+=1

#print(dict(sorted(d.items(), key=lambda item: item[1])))
#print(len([k for k in d.keys() if d[k]>=10]))
dd = [(k,v) for k,v in d.items() if v>=10]
print(len(dd))
# creating bins
d_bins = dict()
i_old = 0
for i in range(1,150):
  ii = i*1000
  d_bins[f"{str(i_old)}-{str(ii)}"] = 0


  # adding elements from d
  for tok, freq in d.items():
    #print(i_old<=freq<ii)
    if i_old<=freq<ii:
      d_bins[f"{str(i_old)}-{str(ii)}"] += 1
  i_old=ii
#print(d_bins)
