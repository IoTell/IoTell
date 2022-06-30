import numpy as np

def zipf(s,Total,Range):
    pdf = []
    for x in range(Range):
        pdf.append((1.0/float(x+1))**s)
    Spdf = sum(pdf)

    pdf = [int(Total*P/Spdf) for P in pdf]
    Arr = []
    idx = 1
    for e in pdf:
        for x in range(e):
            Arr.append(idx)
        idx+=1
    return (pdf,np.array(Arr))
