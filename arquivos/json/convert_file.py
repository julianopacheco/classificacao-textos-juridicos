myFile=open('/Users/julianopacheco/dev/python_workspace/classificacao-textos-juridicos/arquivos/json/crime.json', 'r')
myObject=myFile.read()
u = myObject.encode().decode('utf-8-sig')
myObject = u.encode('utf-8')
myFile.encoding
myFile.close()
myData=json.loads(myObject,'utf-8')
