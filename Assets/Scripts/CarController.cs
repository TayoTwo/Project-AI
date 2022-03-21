using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarController : MonoBehaviour
{
    [Header("Targeting")]
    public Transform target;
    public float rotSpeed;
    public float thrust;
    public float maxSpeed;
    Rigidbody rb;

    // Start is called before the first frame update
    void Start()
    {

        rb = GetComponent<Rigidbody>();
        
    }

    void FixedUpdate(){

        Turn(rotSpeed);
        Thrust(thrust);

        if(rb.velocity.magnitude >= maxSpeed){

            rb.velocity = rb.velocity.normalized * maxSpeed;

        }

    }

    void Thrust(float force){

        rb.AddForce(transform.forward * force);

    }

    void Turn(float s){

        Vector3 targetPos = new Vector3(target.position.x,transform.position.y,target.position.z);

        transform.LookAt(targetPos);

    }

}
