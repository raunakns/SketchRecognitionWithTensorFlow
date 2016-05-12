from generated_proto import sketch_pb2 as Sketch


def convert_array_to_points(points):
    result = []
    for row in points:
        new_point = Sketch.SrlPoint()
        new_point.x = row[0]
        new_point.y = row[1]
        new_point.time = -1
        result.append(new_point)
    return result

def convert_points_to_array(points):
    result = []
    for point in points:
        result.append([point.x, point.y])
    return result
