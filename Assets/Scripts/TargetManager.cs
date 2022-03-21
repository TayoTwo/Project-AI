using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TargetManager : MonoBehaviour
{

    public GameObject targetPrefab;
    public CarController car;
    public List<Target> targets = new List<Target>();

    void Start(){

        car = FindObjectOfType<CarController>();
        SetCarTarget();

    }
    
    public void SpawnTarget(Vector3 pos){

        GameObject obj = (GameObject)Instantiate(targetPrefab,pos,Quaternion.identity);

        targets.Add(obj.GetComponent<Target>());

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
