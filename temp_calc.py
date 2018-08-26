with open("002233.txt", "r") as fp:
    for line in fp.readlines():
       line  = line.split(" ")
       
       print line[0]
       print (float(line[4])+float(line[6]))/2.    #4,6
       print (float(line[5])+float(line[7]))/2.    #4,6
       print ("=======> {}".format((line[11:14])))
       print
       print
       #5,7
