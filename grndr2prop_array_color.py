import json

file = "Allogymnopleuri_#09"
folder = "/home/marvin/Downloads/"

beetle_props = open(file + '_beetle_props.txt', 'a')
ball_props = open(file + '_ball_props.txt', 'a')

with open(folder + file + "/" + file + "_db.grndr") as json_data:
    data = json.load(json_data)

    data_beetle = (-1, (0, 0), (0, 0), False, 0)
    data_ball = (-1, (0, 0), (0, 0), False, 0)

    json_color = ''
    json_ref = 0
    json_x = 0
    json_y = 0
    json_width = 0
    json_heigt = 0
    json_has_direction = False
    json_direction = 0

    for labels in data['Labels']:
        for pool in labels['Label']['ImageBuildPool']:
            for images in pool['Item']['ImageBuilds']:
                json_ref = images['ImageBuild']['ImageReference']
                for layers in images['ImageBuild']['Layers']:
                    for draftitems in layers['Layer']['DraftItems']:
                        for properties in draftitems['DraftItem']['Properties']:
                            if(properties['Property']['ID'] == 'PrimaryColor'):
                                json_color = properties['Property']['Value']
                            if(properties['Property']['ID'] == 'Position'):
                                json_x = properties['Property']['Value'].split(";")[0]
                                json_y = properties['Property']['Value'].split(";")[1]
                            if(properties['Property']['ID'] == 'HasDirection'):
                                json_has_direction = properties['Property']['Value']
                            if(properties['Property']['ID'] == 'Direction'):
                                json_direction = properties['Property']['Value']
                            if(properties['Property']['ID'] == 'Size'):
                                json_width = properties['Property']['Value'].split(";")[0]
                                json_height = properties['Property']['Value'].split(";")[1]
                        #beetle
                        if(json_color == '#00ff00'):
                            data_beetle = (json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction)
                            beetle_props.write(str(data_beetle) + "\n")
                        #ball
                        if(json_color == '#ff0000'):
                            data_ball = (json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction)
                            ball_props.write(str(data_ball) + "\n")
                        #reset
                        json_color == ''
                        #print(data_beetle)
