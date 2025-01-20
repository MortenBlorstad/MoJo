import json

#Read a jsonfile (not really JSON: missing quotes, manually added)
def getObsDict(file):

    #Get file content
    with open(file, 'r') as file:
        inpt = file.read() 
    #Fix single quotes
    inpt = inpt.replace("\'","\"")

    return json.loads(inpt) 
