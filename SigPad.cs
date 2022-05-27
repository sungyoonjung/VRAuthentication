using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SigPad : MonoBehaviour
{

    public Texture2D texture;
    public Vector2 textureSize = new Vector2(50, 100);

    void Start()
    {
        var r = GetComponent<Renderer>();
        texture = new Texture2D((int)textureSize.x, (int)textureSize.y);
        r.material.mainTexture = texture;
        
    }
}
