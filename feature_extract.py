import nltk

def extractAllFeatures(inst):
       
    featureValues = {}

    featureCool(inst, featureValues)
    featureFunny(inst, featureValues)
    featureUseful(inst, featureValues)


def featureCool(featuresFromInstance, featureValues):
    
    coolValue = featuresFromInstance[cool] 
    feat = {isCool: coolValue}

    featureValues.update(feat)

def featureFunny(featuresFromInstance, featureValues):
    
    funnyValue = featuresFromInstance[funny] 
    feat = {isFunny: funnyValue}

    featureValues.update(feat)

def featureUseful(featuresFromInstance, featureValues):
    
    usefulValue = featuresFromInstance[useful] 
    feat = {isUseful: usefulValue}

    featureValues.update(feat)

def featureLength(featuresFromInstance, featureValues):
    
    lengthValue = len(featuresFromInstance[text])
    feat = {length: lengthValue}

    featureValues.update(feat)

