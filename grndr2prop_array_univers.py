import json

file = "Lamarcki_#03"
folder = "data/" + file + "/"

beetle_props = open(folder + file + '_beetle_props.txt', 'a')
ball_props = open(folder + file + '_ball_props.txt', 'a')

with open(folder + file + ".grndr") as json_data:
    data = json.load(json_data)

    data_beetle = []
    data_ball = []

    json_color = ''
    json_ref = 0
    json_x = 0
    json_y = 0
    json_width = 0
    json_heigt = 0
    json_has_direction = False
    json_direction = 0

    json_length = 0

    #get json length
    for images in data['ImageReferences']:
        json_length += 1

    #get json information
    for label in data['Labels']:
        for pool in label['Label']['ImageBuildPool']:
            for ref in pool['Item']['ImageBuilds']:
                json_ref = ref['ImageBuild']['ImageReference']
                for layer in ref['ImageBuild']['Layers']:
                    for draftitem in layer['Layer']['DraftItems']:
                        for prop in draftitem['DraftItem']['Properties']:
                            if(prop['Property']['ID'] == 'PrimaryColor'):
                                json_color = prop['Property']['Value']
                            if(prop['Property']['ID'] == 'Position'):
                                json_x = int(float(prop['Property']['Value'].split(";")[0]))
                                json_y = int(float(prop['Property']['Value'].split(";")[1]))
                            if(prop['Property']['ID'] == 'HasDirection'):
                                if(prop['Property']['Value']=='true'):
                                    json_has_direction =  True
                                else:
                                    json_has_direction =  False
                            if(prop['Property']['ID'] == 'Direction'):
                                json_direction = int(float(prop['Property']['Value']))
                            if(prop['Property']['ID'] == 'Size'):
                                json_width = int(float(prop['Property']['Value'].split(";")[0]))
                                json_height = int(float(prop['Property']['Value'].split(";")[1]))
                        #color labeled method
                        #beetle
                        if(json_color == '#ff0000'):
                            data_beetle.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction))
                        #ball
                        if(json_color == '#00ff00'):
                            data_ball.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction))
                        #two label method
                        if(json_color == '#ffffff'):
                            if(label['Label']['Name'] == 'Beetle'):
                                data_beetle.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction))
                            if(label['Label']['Name'] == 'Ball'):
                                data_ball.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction))
                        #reset
                        json_color = ''

    #sort data
    data_beetle.sort()
    #add missing
    for i in range(0,json_length):
        if not(i in [x[0] for x in data_beetle]):
            data_beetle.insert(i, (i, (0, 0), (0, 0), False, 0))
    #write data
    for x in data_beetle:
        #print(x)
        beetle_props.write(str(x) + "\n")

    #sort data
    data_ball.sort()
    #add missing
    for i in range(0,json_length):
        if not(i in [x[0] for x in data_ball]):
            data_ball.insert(i, (i, (0, 0), (0, 0), False, 0))
    #write data
    for x in data_ball:
        #print(x)
        ball_props.write(str(x) + "\n")
