using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathManager : MonoBehaviour{

    public Transform car;
    public Transform target;
    public List<Node> openSet = new List<Node>();
    public HashSet<Node> closedSet = new HashSet<Node>();
    NodeManager nodeManager;
    TargetManager targetManager;

    void Awake(){

        nodeManager = GetComponent<NodeManager>();
        targetManager = GetComponent<TargetManager>();

        nodeManager.Initialize();
        FindPath(car.position,target.position);

    }

    void Update(){

        //FindPath(car.position,target.position);

    }

    void FindPath(Vector3 s, Vector3 e){

        targetManager.ClearTargets();
        openSet.Clear();
        closedSet.Clear();

        Node start = nodeManager.WorldPosToNode(s);
        Node end = nodeManager.WorldPosToNode(e);

        //Debug.Log(end.gridPos);

        openSet.Add(start);

        while(openSet.Count > 0){

            Node currentNode = openSet[0];

            for(int i = 1; i < openSet.Count;i++){


                //If the open set and closed set have the same F cost then compare their H costs instead (distance to target node)
                if(openSet[i].F() < currentNode.F() 
                || (openSet[i].F() == currentNode.F() && openSet[i].h < currentNode.h)){

                                
                    currentNode = openSet[i];

                }

            }

            openSet.Remove(currentNode);
            closedSet.Add(currentNode);

            if(currentNode == end){

                //We've reached our target
                //Debug.Log("PATH FOUND");
                Retrace(start,end);
                targetManager.SetCarTarget();
                return;

            }


            foreach(Node neigh in nodeManager.GetNeighbours(currentNode)){

                if(!neigh.walkable || closedSet.Contains(neigh)){

                    continue;

                }



                int disToNeighbour = currentNode.g + GetDis(currentNode,neigh);

                if(disToNeighbour < neigh.g || !openSet.Contains(neigh)){

                    neigh.g = disToNeighbour;
                    neigh.h = GetDis(neigh,end);
                    neigh.parent = currentNode;

                    if(!openSet.Contains(neigh)){

                        openSet.Add(neigh);

                    }

                }

            }

        }


    }

    void Retrace(Node start,Node end){

        List<Node> path = new List<Node>();

        Node c = end;

        //Loop backwards
        while(c != start){
            path.Add(c);
            //Spawn a target at the nodes position
            targetManager.SpawnTarget(c.pos);

            //Set the current node to its parent node
            c = c.parent;

        }

        path.Reverse();
        targetManager.targets.Reverse();

    }

    int GetDis(Node a, Node b){

        int x = Mathf.Abs(a.gridPos.x - b.gridPos.x);
        int y = Mathf.Abs(a.gridPos.y - b.gridPos.y);

        if(x > y){

            return 14 * y + 10 * (x-y);

        } else {

            return 14 * x + 10 * (y-x);

        }
         
    }

}
