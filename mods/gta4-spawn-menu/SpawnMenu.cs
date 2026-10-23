// GTA IV Spawn Menu — ScriptHookDotNet (.NET scripting hook for GTA IV / EFLC).
// Положи файл в папку GTAIV\scripts\ — скрипт компилируется сам при запуске игры.
// F7 — открыть/закрыть меню, стрелки вверх/вниз — выбор, Enter — заспавнить.
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using GTA;

public class SpawnMenu : Script
{
    // Одна запись в меню. Чтобы добавить новую вещь — допиши элемент в Items.
    class SpawnItem
    {
        public string Name;
        public Action<SpawnMenu> Spawn;
        public SpawnItem(string name, Action<SpawnMenu> spawn) { Name = name; Spawn = spawn; }
    }

    readonly List<SpawnItem> Items = new List<SpawnItem>();
    bool open;
    int selected;

    public SpawnMenu()
    {
        Items.Add(new SpawnItem("Волосатый мотороллер Короля Британии", m => m.SpawnHairyScooter()));

        KeyDown += OnKeyDown;
        PerFrameDrawing += OnDraw;
    }

    void OnKeyDown(object sender, GTA.KeyEventArgs e)
    {
        if (e.Key == Keys.F7) { open = !open; return; }
        if (!open) return;

        if (e.Key == Keys.Up) selected = (selected + Items.Count - 1) % Items.Count;
        else if (e.Key == Keys.Down) selected = (selected + 1) % Items.Count;
        else if (e.Key == Keys.Enter) { Items[selected].Spawn(this); open = false; }
    }

    void OnDraw(object sender, GraphicsEventArgs e)
    {
        if (!open) return;

        const float x = 40f, y = 120f, rowH = 26f, w = 460f;
        e.Graphics.DrawRectangle(new RectangleF(x, y, w, rowH * (Items.Count + 1) + 8f), Color.FromArgb(180, 0, 0, 0));
        e.Graphics.DrawText("SPAWN MENU  (F7)", x + 10f, y + 4f, Color.Gold);

        for (int i = 0; i < Items.Count; i++)
        {
            float rowY = y + rowH * (i + 1) + 4f;
            if (i == selected)
                e.Graphics.DrawRectangle(new RectangleF(x + 4f, rowY, w - 8f, rowH), Color.FromArgb(160, 200, 16, 46));
            e.Graphics.DrawText(Items[i].Name, x + 12f, rowY + 3f, Color.White);
        }
    }

    // Faggio — штатный мотороллер GTA IV. Красим в цвета британского флага
    // (красный/синий), на руле — водитель в шапке "короля".
    void SpawnHairyScooter()
    {
        Vector3 pos = Player.Character.Position + Player.Character.Direction * 4f;
        Vehicle scooter = World.CreateVehicle(new Model("FAGGIO"), pos);
        if (scooter == null || !scooter.Exists())
        {
            Game.DisplayText("Не удалось заспавнить мотороллер", 3000);
            return;
        }

        scooter.Color = ColorIndex.Red;
        scooter.SpecularColor = ColorIndex.Blue;
        scooter.Heading = Player.Character.Heading;
        scooter.isRequiredForMission = false;

        Game.DisplayText("God save the King! Волосатый мотороллер подан.", 4000);
    }
}
