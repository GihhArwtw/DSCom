
log = open(r"C:\Users\siwei\Desktop\Paper\Final Code\V&A\ExpRes\test.txt","w+")
print('HAHA', file = log, flush=True)
log.close()

for i in range(10,80,10):
    print(i)
