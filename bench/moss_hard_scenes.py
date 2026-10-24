"""Scenes picked to split MOSS-Image from Klein: many small Russian lines, long words, an object count,
attribute binding (each of several look-alike items gets its own colour and label), ordered people and hands
holding lettering. klein_hard's scenes were a tie (2026-10-24); each of these has a failure a reader can see."""
from imaging.ideogram_layout import build_caption, element


def E(desc, box, text=""):
    return element(desc, box, text=text) if text else element(desc, box)


SCENES = {
    "timetable": build_caption("an airport departures board, black panel with amber letters, six rows", [
        E("board header in large amber capitals", [40, 100, 130, 900], "ВЫЛЕТ"),
        E("departure row, amber letters", [180, 60, 270, 940], "Москва 07:15"),
        E("departure row, amber letters", [290, 60, 380, 940], "Казань 08:40"),
        E("departure row, amber letters", [400, 60, 490, 940], "Сочи 09:05"),
        E("departure row, amber letters", [510, 60, 600, 940], "Новосибирск 11:30"),
        E("departure row, amber letters", [620, 60, 710, 940], "Владивосток 13:55"),
        E("departure row, amber letters", [730, 60, 820, 940], "Калининград 16:20"),
        E("small amber footer line", [860, 200, 940, 800], "Терминал B")],
        high_level="A close-up of a departures board: a header, six city rows with times, a footer.",
        aesthetics="clean", lighting="dim terminal light, glowing letters", photo="50mm photo"),
    "jars": build_caption("five glass jars standing in one row on a white kitchen shelf", [
        E("glass jar with red lid filled with rice, white paper label", [300, 20, 750, 190], "РИС"),
        E("glass jar with blue lid filled with buckwheat, white paper label", [300, 210, 750, 380], "ГРЕЧКА"),
        E("glass jar with green lid filled with lentils, white paper label", [300, 400, 750, 570], "ЧЕЧЕВИЦА"),
        E("glass jar with yellow lid filled with sugar cubes, white paper label", [300, 590, 750, 760], "САХАР"),
        E("glass jar with black lid filled with coffee beans, white paper label", [300, 780, 750, 970], "КОФЕ"),
        E("white wooden shelf", [750, 0, 820, 1000])],
        high_level="Exactly five labelled jars, lids red, blue, green, yellow, black from left to right.",
        aesthetics="minimal Scandinavian", lighting="soft daylight", photo="product photo"),
    "runners": build_caption("four runners standing side by side at a finish line, facing the camera", [
        E("runner in a red shirt holding a white card with a big black number at chest height", [200, 20, 950, 240], "1"),
        E("runner in a green shirt holding a white card with a big black number at chest height", [200, 260, 950, 490], "2"),
        E("runner in a blue shirt holding a white card with a big black number at chest height", [200, 510, 950, 740], "3"),
        E("runner in a yellow shirt holding a white card with a big black number at chest height", [200, 760, 950, 980], "4"),
        E("finish banner across the top", [20, 100, 160, 900], "ФИНИШ")],
        high_level="Four runners in red, green, blue, yellow shirts, left to right, holding numbers 1 to 4 in order.",
        aesthetics="sports documentary", lighting="bright overcast", photo="35mm photo"),
    "longword": build_caption("a museum entrance hall with a long carved stone frieze", [
        E("long stone frieze above the doors with carved capital letters", [80, 30, 220, 970], "ДОСТОПРИМЕЧАТЕЛЬНОСТИ"),
        E("small brass plaque beside the door", [420, 760, 520, 960], "Вход свободный"),
        E("yellow warning sign on a stand", [600, 60, 760, 300], "ОСТОРОЖНО! ОКРАШЕНО"),
        E("tall double wooden doors", [260, 340, 900, 660]),
        E("woman in a red coat walking towards the doors, seen from behind", [480, 420, 980, 580])],
        high_level="A museum entrance with a long carved word, a plaque, a warning sign and a visitor.",
        aesthetics="classical", lighting="soft daylight", photo="35mm architecture photo"),
    "card": build_caption("close-up of two hands holding a greeting card in front of a birthday table", [
        E("white greeting card held in both hands, handwritten blue ink, two lines", [250, 200, 650, 800],
          "С днём рождения,\nАлексей!"),
        E("left hand holding the card's left edge, five fingers, thumb in front", [300, 120, 700, 300]),
        E("right hand holding the card's right edge, five fingers, thumb in front", [300, 700, 700, 880]),
        E("birthday cake with exactly seven lit candles", [720, 300, 980, 700])],
        high_level="Two hands hold a handwritten card; behind it a cake with seven candles.",
        aesthetics="warm", lighting="candle glow", photo="50mm close-up"),
    "label": build_caption("a glass bottle of mineral water on a grey stone surface, studio shot", [
        E("brand name on the bottle label in bold blue capitals", [300, 330, 400, 670], "РОДНИК"),
        E("smaller line under the brand", [410, 340, 460, 660], "Минеральная вода"),
        E("volume printed at the bottom of the label", [500, 420, 550, 580], "0,5 л"),
        E("tiny ingredient line at the label bottom", [570, 330, 610, 670], "Состав: вода, кальций, магний"),
        E("clear glass bottle with a blue screw cap", [60, 300, 950, 700])],
        high_level="A product shot of a water bottle whose label has four lines of Russian text of falling size.",
        aesthetics="clean commercial", lighting="softbox studio light", photo="product photo"),
    "chess": build_caption("a wooden chessboard seen from above at an angle, with a few pieces", [
        E("white king standing on the left square", [300, 200, 600, 350]),
        E("black queen standing in the centre", [300, 420, 600, 580]),
        E("white knight lying on its side on the right", [450, 650, 600, 850]),
        E("small paper note on the table edge, handwritten", [800, 300, 920, 700], "Мат в два хода"),
        E("wooden chessboard with 8 by 8 alternating squares", [150, 100, 780, 900])],
        high_level="A chessboard with exactly three pieces: white king left, black queen centre, white knight lying right.",
        aesthetics="moody", lighting="single warm lamp", photo="50mm photo"),
    "newspaper": build_caption("a folded newspaper lying on a cafe table next to a cup of coffee", [
        E("newspaper masthead in black gothic capitals", [80, 100, 190, 800], "ВЕЧЕРНИЙ ГОРОД"),
        E("date line under the masthead, small print", [200, 100, 240, 500], "Пятница, 24 октября"),
        E("bold headline", [260, 100, 380, 800], "Метро откроют до полуночи"),
        E("smaller subheadline", [390, 100, 440, 800], "Новые станции появятся весной"),
        E("cup of black coffee on a saucer", [550, 750, 800, 980])],
        high_level="A newspaper front page with masthead, date, headline and subheadline, coffee beside it.",
        aesthetics="editorial", lighting="morning window light", photo="35mm photo"),
}
