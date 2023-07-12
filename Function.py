def cleanEmptyStroke(allStroks):
    for i in range(len(allStroks)-1):
        if len(allStroks[i]) == 0:
            allStroks.remove(i)
            
    return allStroks