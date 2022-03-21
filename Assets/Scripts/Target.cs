using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Target : MonoBehaviour
{

    public TargetManager targetManager;

    // Start is called before the first frame update
    void Start(){

        targetManager = FindObjectOfType<TargetManager>();
        
    }

    void OnTriggerEnter(Collider c){

        if(c.tag == "Car"){

            int i = targetManager.targets.IndexOf(this);

            c.GetComponent<CarController>().target = targetManager.targets[i + 1].transform;

            targetManager.targets.Remove(this);

            Destroy(gameObject);

        }

    }

}
