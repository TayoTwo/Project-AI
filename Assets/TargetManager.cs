using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TargetManager : MonoBehaviour
{

    public GameObject targetPrefab;
    public CarController car;
    List<Target> targets = new List<Target>();
    public 




    void SpawnTarget(Vector3 pos){

        GameObject obj = (GameObject)Instantiate(targetPrefab,pos,Quaternion.identity);

    }

}
