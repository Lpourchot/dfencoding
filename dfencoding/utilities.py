import pandas as pd
import numpy as np

class dfencoding:
    """dfencoding enables to encode and decode Pandas DataFrames (train w/o test)
    with only one python class instantiation. 
    The goal is to provide methods as easy as possible, with auto filling missing values capabilities.
    Encodings and decodings methods provided are: 

    LABEL ENCODING. 
    TARGET+SMOOTHING ENCODING.
    GET_DUMMIES ENCODING. 
    MINMAX ENCODING.

    Class
    -----

        dfencoding ( train = train dataframe, 
                    target = 'target'name', 
                    test = test dataframe (default = None), 
                    missing_value = Y/N (default = 'N')
                    verbose = (default value = 1),
                    cat_limit = (default value = 100)
                    dummies_limit = (default value = 50)
                   )

        Train, test or data (train + data) encoded or decoded are retrievable by the object attribute .train or .test, .data:

        1 ) Object creation:
        dfe = utilities.dfencoding (train,'target',test)

        2) Choice of encoding(s) method(s):
        dfe.encode()
        ...

        3) Files encoded are retrievable :
        train_encoded = dfe.train
        test_encoded = dfe.test
        both files concatenated_encoded = dfe.data

        dealing with Missing values (missing_value = Y):
         - Replace categories only in train by mode()
         - Replace categories only in test by mode()
         - Replace others nan by mode()

        ****************************** IMPORTANT ************************************
        WHEN USING SPECIFIC COLUMNS LIST (OPTIONAL IN ARGUMENTS OF A METHOD EXCEPT FOR 
        UNDUMMIES) NO CONTROLS ARE MADE. 
        THE USER IS AUTONOMUS TO DETERMINE IF IT IS SMART OR NOT TO ENCODE AND USE SEVERAL
        METHODS (FOR EXAMPLE). IN CERTAIN CASE, NO DECODING COULD BE POSSIBLE.
        UNDUMIES IS ONLY POSSIBLE FOR ALL COLUMNS ENCODED WITH GET_DUMMIED 
        (no columns selection for undummies).
        
        Some controls are made when any columns are mentioned in the method arguments, such as :
        - category type (category / numerical),
        - no target encoding for a column encoded previously by labelencoder,
        - no minmaxencoding for category columns encoded with get_dummies,
        - ...
        
        it is possible to apply consecutively several encoding methods (categorical + minmax) but in order to avoid 
        any issue during decoding, the best practices is to use the correct decode method and in the 
        reverse order of encoding, even if it is planned to work in every order (when columns are not specified in 
        arguments ot the methods, otherwise no control are made).
        Calculation is "very" rapid but you should wait until end of calculation for new action.
  
        
    Parameters
    ----------   
        train:      Training set Dataframe name. 

        target:     Target column name (string) in train : "target".

        test:       Test set Dataframe name.
                    When test file is provided (optional but strongly recommended for categories) 
                    a vertical concatenated file is made (data = train + test) before encoding
                    to perform a global encoding of both train and test (accessible with .data).

        Missing_value:
                    If missing_value = 'Y' (Default ='Y' Else 'N') the missing values are filled with
                    mode() method and the discrepancies between train and test categories are analysed 
                    and fixed.

        verbose:    Enable messages, other values disable messages (default value = 1)

        cat_limit:  isable the start of the method for each categorical column if the number of categories 
                    in files is above the limit (default value = 100). 

        dummies_limit: disable the start of the method for each categorical column if the number of categories 
                    in files is above the limit (default value = 50).   

     Attributes 
    -----------
        train : train file (encoded or decoded)
        test : test file (encoded or decoded)
        data : train + test (encoded or decoded)
    
    Methods 
    -------
        encode(dataframe) / decode(dataframe) :
        - The method encode performs a label encoding of all categorical columns
          and a MinMax encoding for all numerical columns.
        - The previous columns label encoded are included in the MinMax encoding.

        minmaxencode(col=[optional list of columns]) / minmaxdecode(col=[optional list of columns]) :
        - The method minmaxencode performs a numerical encoding :
          (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        Labelencode(col=[optional list of columns]) / Labeldecode(col=[optional list of columns]:
        - The method "labelencode" encodes all categorical columns.
        - The list of columns for the argument "col" limits the encoding or decoding to the list.
        - In case of too many categories (500), the encoding is cancelled.

        targetencode(col=[optional list of columns],weight) / targetdecode(col=[optional list of columns]) :
        - The method "targetencode" encodes all categorical columns with the target.
        - The method "targetdecode" reverses the encoding to the initial values.
        - The target name of the target column has to be provided for encode method.
        - The list of columns for the argument "col" limits the encoding or decoding to the list.
        - The weight (default value = 10000)
        - The calculation is (count * means + weight * mean) / (count + weight)

        get_dummies(col=[optional list of columns]) / undummies()
        - decode reverses the encoding from get_dummies method to the initial values.
        - undummies rverses encoding of all columns encoded by get_dummies (no choice of columns)
        - undummies gives back the initial order of columns in train or test.
    

        Examples 
        --------
            pip install git+https://github.com/Lpourchot/dfencoding.git
            
            from dfencoding import utilities
            
            train = pd.read('train.csv')
            test = pd.read('test.csv')
            dfe = utilities.dfencoding(train, 'target', test, missing_value ='Y')
            dfe.encode()
            dfe.decode()
            train_encoded = dfe.train
            test_encoded = dfe.test
            ...
            dfe.targetencode(["Sex","educ_level"],1000)
            dfe.targetdecode(["Sex","educ_level"])

        Returns
        -------
            data between 0 and 1, except for label encoding but minmaxencode can be applied after the label encoding.
            initial values after decode method    
        """

    def __init__(self, train, target, test = None, missing_value = "Y", verbose = 1, cat_limit = 100, dummies_limit = 50 ):
        
        self.train = train
        self.target = target   
        self.test = test
        self.missing_value = missing_value
        self.verbose = verbose
        self.cat_limit = cat_limit
        self.dummies_limit = dummies_limit         
        self.length_train = len(train)
        self.col_names = list(train.columns)
        self.name_train = "train"
        self.m = {}  # Dict for mapping for minmaxencode.
        self.l = {}  # Dict for mapping for labelencode.
        self.l_inverse = {}  # Dict for mapping for labeldecode.
        self.t = {}  # Dict for mapping for targetencode.
        self.t_inverse = {}  # Dict for mapping for targetdecode.
        self.no_print = "NO"  # avoiding 2 printings for encode method due to the 2 methods used together.
        self.list_col_category = [c for c in self.train.columns if ((self.train[c].dtypes in ["object"]) == True)]
        self.list_col_numeric = [c for c in self.train.columns if (str(self.train[c].dtypes)[0] in ["f","u","i"])== True]
        self.set_dummies = set([x for x in train.columns if (x != self.target) == True])  # set to check encoding capabilities.
        self.set_target = set([x for x in train.columns if (x != self.target) == True])  # set to check encoding capabilities.
        self.set_label = set([x for x in train.columns if (x != self.target) == True])  # set to check encoding capabilities.
        self.set_minmax = set([x for x in train.columns if (x != self.target) == True])  # set to check encoding capabilities.
        self.set_dummies_encoded = set()  # set to check decoding capabilities.
        self.set_target_encoded = set() # set to check decoding capabilities
        self.set_label_encoded = set()  # set to check decoding capabilities
        self.set_minmax_encoded = set()  # set to check decoding capabilities
        
        if (self.verbose == 1) == True :
            intro = "   help(dfencoding) to get information on methods and parameters)   "
            analyse = "  Analyse and filling the missing values   "
            print (intro.center(60, '-'))
            print ()
            if (self.missing_value == "Y") == True:
                print (analyse.center(60, '-'))
                print()
        
        if (self.missing_value == "Y") == True:
            missing_value_col_train = [c for c in self.train.isnull().sum().index if (self.train.isnull().sum()[c] != 0) == True] 
            missing_train = self.train.isnull().sum().sum()
            if ((missing_train != 0) & (self.verbose == 1)) == True:  # Check of the missing values in train
                print("Total of {} missing values train, in columns : {}".format(missing_train,missing_value_col_train))
                print("will be replaced by the value that appears most often in the according columns")
                print()
            else:
                if (self.verbose == 1):
                    print("No missing value in train")
                    print()
            if self.test is not None :  # Check the missing values in test
                missing_test = self.test.isnull().sum().sum()
                missing_value_col_test = [c for c in self.test.isnull().sum().index if (self.test.isnull().sum()[c] != 0) == True] 
                if ((missing_test != 0) & (self.verbose == 1)) == True :
                    print("Total of {} missing values test, in columns : {}".format(missing_test,missing_value_col_test))
                    print("will be replaced by the value that appears most often in the according columns")
                    print()
                else:
                    if (self.verbose == 1) == True:
                        print("No missing value in test")
                        print()
            
            for col in self.list_col_category:  # Handle missing values in categories.
                if (col == self.target) == False:
                    if self.test is not None :
                        train_only = list(set(self.train[col].unique()) - set(self.test[col].unique()))  # Categories only in train.
                        if ((len(train_only) > 0) & (self.verbose == 1)) == True :
                            print("The column '{}' includes category(ies) only in train, replaced by the largest column category in train".format(col))
                            print()
                        test_only = list(set(self.test[col].unique()) - set(self.train[col].unique()))  # Categories only in test.
                        if ((len(test_only) > 0) & (self.verbose == 1)) == True :
                            print("The column '{}' includes category(ies) only in test, replaced by the largest column category in train".format(col))
                            print()
                        both = list(set(self.test[col].unique()).union(set(self.train[col].unique())))  # Categories in both.
                        self.train.loc[self.train[col].isin(train_only), col] = np.nan  # Replace categories only in train by nan.
                        self.test.loc[self.test[col].isin(test_only), col] = np.nan  # Replace categories only in test by nan.
                        mode = self.train[col].mode().values[0]  # Catch the largest category in train.
                        self.train[col] = self.train[col].fillna(mode)  # Fill the nan in train by the largest train category.
                        self.test[col] = self.test[col].fillna(mode)  # Fill the nan in test by the largest train category.
                    else :
                        mode = self.train[col].mode().values[0]  # Catch the largest category in train.
                        self.train[col] = self.train[col].fillna(mode)  # Fill the nan in train by the largest train category.  
            
            for col in self.list_col_numeric:  # Handle missing values in numeric columns.
                if (col == self.target) == False:
                    median_train = self.train[col].median()  # Catch the median in train.
                    self.train[col] = self.train[col].fillna(median_train)  # Fill the nan in train by the median train value.
                    if self.test is not None :
                        median_test = self.test[col].median()  # Catch the median in test.
                        self.test[col] = self.test[col].fillna(median_test)  # Fill the nan in test by the median train value.

        
        if (self.test is not None) == True :
            self.length_test = len(self.test)
            self.name_test = "test"
            self.test[target] = np.nan
            self.data = pd.concat([self.train, self.test]).reset_index(drop=True)
        else :
            self.data = self.train 
        self.length_data = len(self.data)
    
    def printing(self):
        end_train = "TRAIN"
        end_test = "TEST"
        end = " to retreive train or test, if you instantied dfe = utilities.dfencoding(train,'target') => train = dfe.train or test = dfe.test "
        print(end.center(60, '-'))
        print()
        print(end_train.center(60, '-'))
        display(self.train.head(5))
        if self.test is not None :
            print(end_test.center(60, '-'))
            display(self.test.head(5))
    
    def encode(self):
        self.no_print = "YES"
        self.labelencode()
        self.no_print = "NO"
        self.minmaxencode()
    
    def decode(self):
        self.no_print = "YES"
        self.minmaxdecode()   
        self.no_print = "NO"
        self.labeldecode() 
    
    def minmaxencode(self, col = None):
         
        self.list_col_numeric = [ x for x in list(self.set_minmax) if (str(self.train[x].dtypes)[0] in ["f","u","i"])== True]
        self.col = col if col is not None else self.list_col_numeric   
        
        for c in self.col:
            min_col = self.data[c].min()
            max_col = self.data[c].max()
            diff = max_col - min_col
            name = self.name_train + c  # we choose Dataframe's name of train (same column name than test).
            self.m[name]=dict(zip(['min','max'],[min_col,max_col]))  # dict (key =col name, values subdict) subdict: {min : .., max:..}.
            df = self.data[c].transform(lambda x :(x - min_col)/(diff) )  # MinMax calculation.
            self.data[c] = df      
            self.set_minmax_encoded.add(c)  # Flag to enable decoding capability for minmaxdecode.
            self.set_minmax.discard(c)

        if (self.test is not None) == True :
            self.test = self.data.iloc[self.length_test:,:].copy()
            self.test.drop(self.target, axis = 1, inplace = True)
            self.train = self.data.iloc[:self.length_train,:]
        else :
            self.train = self.data
        
        if ((self.no_print == "NO") & (self.verbose == 1)) == True:
            self.printing()

    def minmaxdecode(self, col = None):
       
        self.col = set(col) if col is not None else self.set_minmax_encoded  
        
        for c in self.col:
            a = self.name_train + c
            min_col = self.m[a]['min']
            max_col = self.m[a]['max']                
            diff = max_col - min_col  
            df = self.data[c].transform(lambda x : x * (max_col - min_col) + min_col)
            self.data[c] = df
            self.set_minmax.add(c)  # Flag to enable minmaxencode
        self.set_minmax_encoded.difference_update(self.col)  # Flag to disable capability for decoding minmaxdecode.
        
        if (self.test is not None) == True :
            self.test = self.data.iloc[self.length_test:,:].copy()
            self.test.drop(self.target, axis = 1, inplace = True)
            self.train = self.data.iloc[:self.length_train,:]
        else :
            self.train = self.data
        
        if ((self.no_print == "NO") & (self.verbose == 1)) == True:
             self.printing()
                                                                                                                                                                                
    
    def labelencode(self, col=None):
            
        self.list_col_category = [x for x in list(self.set_label) if self.data[x].dtypes in ["object"]]
        self.col = col if col is not None else self.list_col_category 

        for c in self.col:  
            self.unik = pd.unique(self.data[c])
            len_unik = len(self.unik)
            if (len_unik > self.cat_limit) == False:  # Check the number of unique value.
                self.length = len(self.data[c])
                num = []
                label = []   
                for i in range(0, len(self.unik)):
                               num.append(i)
                               label.append(self.unik[i])
                name = self.name_train + c  # Concatenation of dataframe's name and column for the mapping table'name.         
                self.l[name] = dict(zip(label,range(0,self.length_data)))    # Mapping dict for encoding.
                self.l_inverse[name] = dict(zip(range(0,self.length_data),label))   # Mapping dict for decoding.
                self.data[c] = self.data[c].map(self.l[name])
                self.set_label_encoded.add(c)  # Flag to enable decoding capability for labeldecode.

            else :
                if (self.verbose == 1) == True:
                    print("The categorical column '{}' is above the limit of {} categories with {} categories and won't be encoded".format(c,self.cat_limit,len_unik))
                    print()

        if (self.test is not None) == True :
            self.test = self.data.iloc[self.length_test:,:].copy()
            self.test.drop(self.target, axis = 1, inplace = True)
            self.train = self.data.iloc[:self.length_train,:]
        else :
            self.train = self.data

        if ((self.no_print == "NO") & (self.verbose == 1)) == True:
            self.printing()
                                                                                                                                                                                                        
    def labeldecode(self, col = None):
        
        self.col = set(col) if col is not None else self.set_label_encoded
        iterat = self.col.copy()
        for c in [x for x in iterat if (x not in self.set_minmax_encoded) == True]:
            a = self.name_train + c
            self.data[c] = self.data[c].astype('int').map(self.l_inverse[a])  # Map the encoding.
            self.set_label_encoded.discard(c)  # Flag to disable capability for decoding labeldecode
        
        if (self.test is not None) == True :
            self.test = self.data.iloc[self.length_test:,:].copy()
            self.test.drop(self.target, axis = 1, inplace = True)
            self.train = self.data.iloc[:self.length_train,:]
        else :
            self.train = self.data

        if ((self.no_print == "NO") & (self.verbose == 1)) == True:
            self.printing()

    def targetencode(self, col = None, weight = 10000):
        
        self.list_col_category = [x for x in list(self.set_target) if self.data[x].dtypes in ["object"]]
        self.col = col if col is not None else self.list_col_category 
        self.weight = weight
        self.mean = self.data[self.target].mean()

        for c in self.col:
            self.unik = pd.unique(self.data[c])
            len_unik = len(self.unik)
            if (len_unik > self.cat_limit) == False:
                name = self.name_train + c   # Concatenation of dataframe's name and column for the mapping table'name.       
                agg = self.data.groupby(c)[self.target].agg(["count", "mean"])  # Calculation of the value per category.
                count = agg["count"]
                means = agg["mean"]
                agg['smooth'] = (count * means + self.weight * self.mean) / (count + self.weight)                          
                self.t[name] = dict(zip(agg.index.values,agg["smooth"].values))    # Mapping dict for encoding.
                self.t_inverse[name] = dict(zip(agg["smooth"].values,agg.index.values))   # Mapping dict for decoding.
                self.data[c] = self.data[c].map(self.t[name])
                self.set_target_encoded.add(c)  # Flag to enable decoding capability for targetdecode.
             
               
            else :
                if (self.verbose == 1) == True:
                    print("The categorical column '{}' is above the limit of {} categories with {} categories and won't be encoded".format(c,self.cat_limit,len_unik))
                    print()
        
        if (self.test is not None) == True :
            self.test = self.data.iloc[self.length_test:,:].copy()
            self.test.drop(self.target, axis = 1, inplace = True)
            self.train = self.data.iloc[:self.length_train,:]
        else :
            self.train = self.data
        
        if (self.verbose == 1) == True:
            self.printing()
            

    def targetdecode(self, col = None):
                      
        self.col = set(col) if col is not None else self.set_target_encoded
  
        iterat = self.col.copy()
        for c in [x for x in iterat if (x not in self.set_minmax_encoded) == True]:
            a = self.name_train + c
            self.data[c] = self.data[c].map(self.t_inverse[a])  # Map the encoding.
            self.set_target_encoded.discard(c)  # Flag to disable capability for decoding targetdecode
            
        if (self.test is not None) == True :
            self.test = self.data.iloc[self.length_test:,:].copy()
            self.test.drop(self.target, axis = 1, inplace = True)
            self.train = self.data.iloc[:self.length_train,:]
        else:
            self.train = self.data
        
        if (self.verbose == 1) == True:
            self.printing() 
    
    def get_dummies(self, col = None):  
        
        self.list_col_category = [x for x in list(self.set_dummies) if self.data[x].dtypes in ["object"]]
        self.col = col if col is not None else self.list_col_category 
        
        for c in self.col:
            self.unik = pd.unique(self.data[c])
            len_unik = len(self.unik)
            if (len_unik > self.dummies_limit) == False:
                df = pd.get_dummies(self.data[c],prefix=c)
                
                self.data = pd.concat([self.data,df],axis=1)
                self.data.drop([c], axis = 1, inplace = True)                
                self.set_dummies_encoded.add(c)  # Flag to enable decoding capability for undummies.
                self.set_minmax.discard(c)  # Flag to avoid encoding capability for minmaxencode.
                self.set_label.discard(c)  # Flag to avoid encoding capability for labelencode.
                self.set_target.discard(c)  # Flag to avoid encoding capability for targetencode.
            else :
                if (self.verbose == 1) == True:
                    print("The categorical column '{}' is above the limit of {} categories with {} categories and won't be encoded".format(c,self.dummies_limit,len_unik))
                    print()
                    
        if (self.test is not None) == True :
            self.test = self.data.iloc[self.length_test:,:].copy()
            self.test.drop(self.target, axis = 1, inplace = True)
            self.train = self.data.iloc[:self.length_train,:]
        else:
            self.train = self.data
        
        if (self.verbose == 1) == True:
            self.printing()
                  
    def undummies(self):
        
        iterat = self.set_dummies_encoded.copy()
        for c in iterat:
            cols2collapse = {item.split(c)[0]: (c in item) for item in self.data.columns}
            series_list = []
            
            for col, needs_to_collapse in cols2collapse.items():
                if needs_to_collapse:
                    undummified = (
                        self.data.filter(like=c+"_")
                        .idxmax(axis=1)
                        .apply(lambda x: x.split(c+"_", maxsplit=1)[1])
                        .rename(col))
                    series_list.append(undummified)
                else:
                    series_list.append(self.data[col])         
            self.data = pd.concat(series_list, axis=1)
            self.data.rename(columns={"": c},inplace= True)     
            self.set_minmax.add(c)  # Flag to enable encoding for minmax.
            self.set_label.add(c)  # Flag to enable encoding for label.
            self.set_target.add(c)  # Flag to enable encoding for target.
            self.set_dummies_encoded.discard(c)  # Flag to disable capability for decoding get_dummies.

        if (self.test is not None) == True :
            self.test = self.data.iloc[self.length_test:,:].copy()
            self.test.drop(self.target, axis = 1, inplace = True)
            self.train = self.data.iloc[:self.length_train,:]
        else:
            self.train = self.data
        
        
        if (self.verbose == 1) == True:
            self.printing()
        
