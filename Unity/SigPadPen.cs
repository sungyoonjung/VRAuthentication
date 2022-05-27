using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using System.Text;
using System.IO;
using System;
//using System.Diagnostics;
using System.Threading;

public class SigPadPen : MonoBehaviour
{

    [SerializeField] private Transform _tip;
    // set size of pixel array that is drawn with the pen
    [SerializeField] private Transform _deathray;
    [SerializeField] private Transform _sigpadplane;
    [SerializeField] private Text _depthtext;
    [SerializeField] private Text _resultstext;
    [SerializeField] private int _penSize = 5;
    [SerializeField] private float _updateIncrement = 0.01f;

    private Renderer _renderer;
    private Color[] _colors;
    private float _tipHeight;

    private int _filectr;
    private int _xdim;
    private int _ydim;

    private Renderer _thisRend;
    private RaycastHit _touch;
    private Vector3 _hitPosition;
    private SigPad _sigpad;
    private Vector2 _touchPos, _lastTouchPos;
    private bool _touchedLastFrame;
    private Quaternion _lastTouchRot;
    private int _depth;

    private Color[] _clearcolors;
    private RaycastHit _touch2;
    float _hitDistance;
    private int[,] _pixelArray;

    // Start is called before the first frame update
    void Start()
    {
        _xdim = 50;
        _ydim = 100;
        _filectr = 1;

        // need access to the material color of the tip
        _renderer = _tip.GetComponent<Renderer>();
        // makes an array that is pensize^2 long, each entry is the pentip color
        _colors = Enumerable.Repeat(_renderer.material.color,  _penSize * _penSize).ToArray();
        _tipHeight = _tip.localScale.y;

        //_thisRend = _sigpadplane.GetComponent<Renderer>();
        //_thisRend.material.SetColor("_Color", Color.red);

        // initialize pixel array
        _pixelArray = new int[_xdim, _ydim];
        ClearArray();
        Debug.Log("Hello World");
        
    }

    // Update is called once per frame
    void Update()
    {
        Draw();
    }

    public void ClearPad()
    {
        _thisRend = _sigpadplane.GetComponent<Renderer>();
        _sigpad = _sigpadplane.GetComponent<SigPad>();
        _clearcolors = Enumerable.Repeat(_thisRend.material.color, 1).ToArray();
        
        // DEBUG
        //Color[] pixels = _sigpad.texture.GetPixels();
        //Debug.Log("number of pixels: " + pixels.Length);

        // go over all pixels and reset color.
        for (int i = 0; i < _xdim; i++)
        {
            for (int j = 0; j < _ydim; j++)
            {
                _sigpad.texture.SetPixels(i, j, 1, 1, _clearcolors);
            }
        }
        
        _sigpad.texture.Apply();

        ClearArray();
    }

    private void ClearArray()
    {
        for (int i = 0; i < _xdim; i++)
        {
            for (int j = 0; j < _ydim; j++)
            {
                _pixelArray[i, j] = 0;
            }
        }
    }

    public void SubmitSignature()
    {
        // string _filename = "Signature_" + DateTime.Now.ToString("yyyy.MM.dd_hh.mm.ss") + ".txt";
        string _filename = "testsig" + _filectr.ToString() + ".txt";
        //_filectr++;
        string[] textdumppaths = { @"C:\Users", "SYJ", "Documents", "Code", "Unity", "VRAuth", "textfiles", _filename };
        //string[] textdumppaths = { @"G:\My Drive", "Colab Notebooks", "VR_auth", "text files", _filename };
        string textdumpFullPath = Path.Combine(textdumppaths);

        StreamWriter writer = new StreamWriter(textdumpFullPath, true); //Application.persistentDataPath
        writer.AutoFlush = true;
        
        for (int i = 0; i < _xdim; i++)    // for some reason, line 46 cuts off in the middle
        {
            //Debug.Log("i = " + i);
            for (int j = 0; j < _ydim; j++)
            {
                writer.Write(_pixelArray[i, j].ToString());
                // prevent trailing comma at the end of a row
                if (j < _ydim - 1)
                {
                    writer.Write(",");
                }
                //Debug.Log("["+i+","+j+"]");
            }
            writer.Write("\n"); // new line after each row
        }

        writer.Close();

        Debug.Log("starting script");
        string strCmdText;
        strCmdText = "/K C:/Users/SYJ/AppData/Local/Programs/Python/Python39/python.exe C:/Users/SYJ/Documents/Code/Python/runColab/main.py";
        //strCmdText = "/C notepad.exe";
        System.Diagnostics.Process.Start("cmd.exe", strCmdText);

        Thread.Sleep(5000);

        Debug.Log("ending script");

        string[] resultspaths = { @"C:\Users", "SYJ", "Documents", "Code", "Unity", "VRAuth", "textfiles", "results", "results.txt" };
        string resultsFullPath = Path.Combine(resultspaths);
        StreamReader reader = new StreamReader(resultsFullPath, true);
        _resultstext.text = reader.ReadToEnd();

    }

    private void Draw() 
    {
        // CONDITIONAL TO WRITE - marker is within tipHeight from the pad
        if (Physics.Raycast(_tip.position, transform.up, out _touch, _tipHeight))
        {
            // if depth ray hits the pad... check if it's the pad, then calc depth
            if (Physics.Raycast(_deathray.position, transform.up, out _touch2))
            {
                if (_touch2.transform.CompareTag("SigPad"))
                {
                    _hitDistance = Mathf.Floor((_touch2.distance - 0.2000000f) * 1000f + 200);
                    _depth = 255 - (int)_hitDistance;
                    if (_depth < 0) _depth = 0;
                    _depthtext.text = "Depth = " + _depth;
                }
            }

            // if marker ray hit the pad
            if (_touch.transform.CompareTag("SigPad"))
            {
                // check if cache already has sigpad, populate
                if (_sigpad == null)
                {
                    _sigpad = _touch.transform.GetComponent<SigPad>();
                }

                // get the position on the pad to write on
                _touchPos = new Vector2(_touch.textureCoord.x, _touch.textureCoord.y);

                // converting the 2048x2048 position to a screen position
                // _touchPos.x,y is a value between 0 and 2047
                // x: 0-_xdim, y: 0-_ydim
                var x = (int)(_touchPos.x * _sigpad.textureSize.x - (_penSize / 2));
                var y = (int)(_touchPos.y * _sigpad.textureSize.y - (_penSize / 2));

                // if pixels are outside of sigpad, exit
                if (y < 0 || y > _sigpad.textureSize.y || x < 0 || x > _sigpad.textureSize.x) return;

                // Debug.Log("(" + x + ", " + y + ")");
                _pixelArray[x, y] = _depth;

                if (_touchedLastFrame)
                {

                    //Debug.Log("nice");

                    _sigpad.texture.SetPixels(x, y, _penSize, _penSize, _colors);

                    // loop to determine paint frequency of interpolated painting
                    // in other words, painting in the lines. 
                    for (float f = 0.01f; f < 1.00f; f += _updateIncrement)
                    {
                        var lerpX = (int)Mathf.Lerp(_lastTouchPos.x, x, f);
                        var lerpY = (int)Mathf.Lerp(_lastTouchPos.y, y, f);
                        _sigpad.texture.SetPixels(lerpX, lerpY, _penSize, _penSize, _colors);
                    }

                    // lock pen rotation so it doesn't snap up due to physics when approaching pad
                    transform.rotation = _lastTouchRot;

                    _sigpad.texture.Apply();
                }

                // set cache for next frame
                _lastTouchPos = new Vector2(x, y);
                _lastTouchRot = transform.rotation;
                _touchedLastFrame = true;
                return;
            }
        }


        // reset sigpad and frame touch each Draw cycle
        _sigpad = null;
        _touchedLastFrame = false;
    }   
}
