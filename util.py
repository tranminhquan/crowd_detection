from datetime import datetime
from shapely.geometry import LineString


def perpendicular_line(line):
    """Return a perpendicular line to the given line."""
    x1, y1, x2, y2 = line

    a = (x1, y1)
    b = (x2, y2)
    cd_length = 30

    ab = LineString([a, b])
    left = ab.parallel_offset(cd_length / 2, 'left')
    right = ab.parallel_offset(cd_length / 2, 'right')
    c = left.boundary[1]
    d = right.boundary[0]  # note the different orientation for right offset
    #convert left and right to list
    to_list = lambda x: [int(item) for sublist in list(x.coords) for item in sublist]
    c = to_list(left)
    d = to_list(right)
    e = d[-2:]
    e.extend(c[:2])
    print(c,d,e)
    return c,d,e
            
def timestamp2datetime(s):
   return datetime.fromtimestamp(int(s)).strftime("%Y-%m-%d %H:%M:%S")
