import numpy as np
class one_hot:
    def onehotencode(df,column,prefix):
        keys=np.unique(df[column])
        names=[]
        for x in keys:
            if x in df:
                print ("Already Exists")
                continue
            names.append(str(prefix)+str(x))
            df[str(prefix)+str(x)]=np.where(df[column]==x,1.0,0.0)
        return df,names