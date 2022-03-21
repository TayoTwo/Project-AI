using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TargetManager : MonoBehaviour
{

    public GameObject targetPrefab;
    public CarController car;
    public List<GameObject> targets = new List<GameObject>();

    void Start(){

        car = FindObjectOfType<CarController>();
        SetCarTarget();

    }
    
    public void SpawnTarget(Vector3 pos){

        //Spawn our target prefab at a specific position
        GameObject obj = (GameObject)Instantiate(targetPrefab,pos,Quaternion.identity);

        targets.Add(obj);

    }

    public void SetCarTarget(){

        car.target = targets[0].transform;

    }

    public void ClearTargets(){

        for(int i = 0; i < targets.Count;i++){

            Destroy(targets[i].gameObject);

        }

        targets.Clear();

    }

}
