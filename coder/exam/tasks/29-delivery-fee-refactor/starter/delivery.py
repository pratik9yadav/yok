def calc(d, amt, exp, mem, hr):
    # d = km, amt = order value (paise), exp = express?, mem = member?, hr = hour of day (0-23)
    if d < 0:
        raise ValueError("bad distance")
    if d > 25:
        raise ValueError("too far")
    if amt >= 50000 and exp == False:
        return 0
    if mem == True:
        if d <= 3:
            f = 0
        else:
            if d <= 7:
                f = 1500
            else:
                f = 1500 + int((d - 7) * 500)
        if exp == True:
            f = f + 3000
        if hr >= 22 or hr < 6:
            f = f + 1000
    else:
        if d <= 3:
            f = 2000
        else:
            if d <= 7:
                f = 3500
            else:
                f = 3500 + int((d - 7) * 500)
        if exp == True:
            f = f + 3000
        if hr >= 22 or hr < 6:
            f = f + 2500
    if f > 15000:
        f = 15000
    return f
